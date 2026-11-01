#!/usr/bin/env python3
# Pack an extracted RomFS folder into one image file the host reads in place of the folder (hle_fs.c, "RomFS image"): DM_ROMFS may name the image instead of a folder.
# One file instead of hundreds of thousands copies to an iPhone at full speed (Tears of the Kingdom's 304,000 files ran the phone's file service out of open files), and the host
# keeps one descriptor open for the whole RomFS.
# Layout (little-endian):
#   header  32 bytes: magic "DMROMFS1", u64 index offset, u64 index size, u64 entry count
#   data    each file's bytes, one after another
#   index   entry count entries of 24 bytes: u64 a, u64 b, u32 name offset, u16 name length, u8 type (0 directory, 1 file), u8 0
#           a directory: a = its first child's entry, b = its number of children; a file: a = data offset in the image, b = size
#           then the names, one pool. Entry 0 is the root (empty name). Each directory's children are consecutive entries sorted by name (bytes), so a lookup is a
#           binary search per path component; directories are numbered breadth-first.
# The index comes last, so a copy cut off partway has no valid index (the header points past the end of the file) and the host refuses it.
# Names are the guest names: a host name chosen on a case collision (.dm-case-N) is replaced by the one in its xattr, as hle_fs.c's guest_name does.
# Usage: romfs_image.py [--reuse] <RomFS folder> <out image>   (written to <out>.part, renamed at the end)
#        romfs_image.py --overlay [--reuse] <base image> <overlay folder> <out image>
# --reuse checks the source tree's names and file metadata, the packer's version and the image's identity before keeping a previous image.
import argparse, hashlib, json, os, struct, subprocess
from pathlib import Path

MAGIC = b'DMROMFS1'
HEAD = struct.Struct('<8sQQQ')
ENT = struct.Struct('<QQIHBB')
GUEST_NAME = 'org.d2rmobile.guest-name'


def guest(parent, name):
    if not name.startswith('.dm-case-'): return name
    r = subprocess.run(['xattr', '-p', GUEST_NAME, os.path.join(parent, name)], capture_output=True, text=True)
    return r.stdout.rstrip('\n') if r.returncode == 0 else name


def children(path):
    out = []
    with os.scandir(path) as it:
        for e in it:
            out.append((guest(path, e.name).encode(), e.path, e.is_dir(follow_symlinks=True)))
    out.sort()
    return out


def entries(src):
    ents = [[b'', src, True, 0, 0]]   # name, host path, is dir, a, b
    i = 0
    while i < len(ents):   # breadth-first: each directory's children land as one consecutive run
        name, path, isdir, _, _ = ents[i]
        if isdir:
            kids = children(path)
            ents[i][3], ents[i][4] = len(ents), len(kids)
            ents.extend([n, p, d, 0, 0] for n, p, d in kids)
        i += 1
    return ents


def file_identity(path):
    s = os.stat(path)
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def source_digest(src, ents):
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(os.fsencode(os.path.realpath(src)) + b'\0')
    for name, path, isdir, _, _ in ents:
        h.update(struct.pack('<I?', len(name), isdir)); h.update(name)
        h.update(json.dumps(file_identity(path)).encode() + b'\0')
    return h.hexdigest()


def pack(src, dst, reuse=False):
    src, dst = os.fspath(src), os.fspath(dst)
    ents = entries(src)
    digest = source_digest(src, ents)
    stamp = dst + '.inputs.json'
    if reuse:
        try:
            cached = json.loads(Path(stamp).read_text())
            if cached == {'source': digest, 'image': file_identity(dst)}: return len(ents)
        except (OSError, ValueError): pass
    part = dst + '.part'
    with open(part, 'wb') as out:
        out.write(b'\0' * HEAD.size)
        off = HEAD.size
        for e in ents:
            if e[2]: continue
            with open(e[1], 'rb') as f:
                n = 0
                while True:
                    b = f.read(8 << 20)
                    if not b: break
                    out.write(b); n += len(b)
            e[3], e[4] = off, n; off += n
        pool, idx = bytearray(), bytearray()
        for name, _, isdir, a, b in ents:
            idx += ENT.pack(a, b, len(pool), len(name), 0 if isdir else 1, 0); pool += name
        out.write(idx); out.write(pool)
        out.seek(0); out.write(HEAD.pack(MAGIC, off, len(idx) + len(pool), len(ents)))
    # A changing source must not publish a mixed image or a reuse stamp for stale data.
    if source_digest(src, entries(src)) != digest:
        os.unlink(part)
        raise RuntimeError('RomFS changed while packing; run the packer again when its files are stable')
    os.replace(part, dst)
    Path(stamp + '.part').write_text(json.dumps({'source': digest, 'image': file_identity(dst)}))
    os.replace(stamp + '.part', stamp)
    return len(ents)


def image_entries(img):
    """Return the image as {guest path: (is_dir, offset, size)} after validating its index."""
    return read_index(img)


def overlay_image(base, overlay, dst, reuse=False):
    """Build dst from base while replacing/adding files found under overlay.

    Unchanged files are streamed directly from the existing image.  The base image may therefore
    be many gigabytes without ever being expanded into hundreds of thousands of loose files.
    """
    base, overlay, dst = map(os.fspath, (base, overlay, dst))
    overlay_ents = entries(overlay)
    digest = {
        'tool': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base': file_identity(base),
        'overlay': source_digest(overlay, overlay_ents),
    }
    stamp = dst + '.overlay.inputs.json'
    if reuse:
        try:
            cached = json.loads(Path(stamp).read_text())
            if cached == {'source': digest, 'image': file_identity(dst)}: return len(image_entries(dst))
        except (OSError, ValueError, AssertionError, struct.error): pass

    # path -> None for a directory, ('image', offset, size) or ('file', path) for a file.
    nodes = {'': None}
    for path, (isdir, off, size) in image_entries(base).items():
        nodes[path] = None if isdir else ('image', off, size)

    def remove_at(path):
        prefix = path + '/'
        for old in [p for p in nodes if p == path or p.startswith(prefix)]: nodes.pop(old, None)

    def add_overlay(path, host, isdir):
        if path:
            parent = path.rpartition('/')[0]
            if nodes.get(parent, None) is not None: remove_at(parent)
            nodes[parent] = None
        if isdir:
            if path in nodes and nodes[path] is not None: remove_at(path)
            nodes[path] = None
            for name, child, child_dir in children(host):
                child_path = (path + '/' if path else '') + os.fsdecode(name)
                add_overlay(child_path, child, child_dir)
        else:
            remove_at(path)
            nodes[path] = ('file', host)

    for name, host, isdir in children(overlay): add_overlay(os.fsdecode(name), host, isdir)

    children_by_dir = {p: [] for p, source in nodes.items() if source is None}
    for path in nodes:
        if not path: continue
        parent, _, name = path.rpartition('/')
        children_by_dir.setdefault(parent, []).append((os.fsencode(name), path))
    for kids in children_by_dir.values(): kids.sort(key=lambda x: x[0])

    # Breadth-first numbering keeps every directory's children in one consecutive run, as the host expects.
    built = [[b'', '', None, 0, 0]]   # name, path, source, a/offset, b/size
    i = 0
    while i < len(built):
        _, path, source, _, _ = built[i]
        if source is None:
            kids = children_by_dir.get(path, [])
            built[i][3], built[i][4] = len(built), len(kids)
            built.extend([name, child, nodes[child], 0, 0] for name, child in kids)
        i += 1

    part = dst + '.part'
    with open(base, 'rb') as old, open(part, 'wb') as out:
        out.write(b'\0' * HEAD.size); off = HEAD.size
        for e in built:
            source = e[2]
            if source is None: continue
            kind, where, *length = source
            if kind == 'image':
                old.seek(where); left = length[0]
                while left:
                    chunk = old.read(min(left, 8 << 20))
                    if not chunk: raise ValueError(f'incomplete base RomFS image: {base}')
                    out.write(chunk); left -= len(chunk)
                size = length[0]
            else:
                size = 0
                with open(where, 'rb') as src:
                    while True:
                        chunk = src.read(8 << 20)
                        if not chunk: break
                        out.write(chunk); size += len(chunk)
            e[3], e[4] = off, size; off += size
        pool, idx = bytearray(), bytearray()
        for name, _, source, a, b in built:
            idx += ENT.pack(a, b, len(pool), len(name), 0 if source is None else 1, 0); pool += name
        out.write(idx); out.write(pool)
        out.seek(0); out.write(HEAD.pack(MAGIC, off, len(idx) + len(pool), len(built)))

    if source_digest(overlay, entries(overlay)) != digest['overlay']:
        os.unlink(part)
        raise RuntimeError('RomFS overlay changed while packing; run the packer again when its files are stable')
    os.replace(part, dst)
    Path(stamp + '.part').write_text(json.dumps({'source': digest, 'image': file_identity(dst)}))
    os.replace(stamp + '.part', stamp)
    return len(built)


def read_index(img):   # for tests and checks: {guest path: (is dir, offset, size)}
    with open(img, 'rb') as f:
        magic, ioff, isz, n = HEAD.unpack(f.read(HEAD.size))
        assert magic == MAGIC
        f.seek(ioff); raw = f.read(isz)
    pool = raw[n * ENT.size:]
    e = [ENT.unpack_from(raw, k * ENT.size) for k in range(n)]
    out, todo = {}, [(0, '')]
    while todo:
        k, path = todo.pop()
        a, b, no, nl, t, _ = e[k]
        if t == 0:
            if path: out[path] = (True, 0, 0)
            todo.extend((c, (path + '/' if path else '') + pool[e[c][2]:e[c][2] + e[c][3]].decode()) for c in range(a, a + b))
        else: out[path] = (False, a, b)
    return out


ID_MAGIC = 'DMROMFSID1'
_FNV_OFFSET, _FNV_PRIME, _MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1


def _clean(text):   # one token of the identity line: no spaces, no newlines
    return ''.join(c if c.isalnum() or c in '._-+' else '_' for c in str(text)) or '-'


def identity(img, title='', version='', build=''):
    """The identity line of a complete image (dm/host/romfs_identity.h defines the format and the C reader): its size and an FNV-1a 64 hash of its header and index
    (the file data is not read, so a 14 GB image takes a moment). Raises ValueError for a cut off or invalid image."""
    size = os.path.getsize(img)
    with open(img, 'rb') as f:
        head = f.read(HEAD.size)
        if len(head) != HEAD.size: raise ValueError(f'not a RomFS image: {img}')
        magic, ioff, isz, _ = HEAD.unpack(head)
        if magic != MAGIC or ioff < HEAD.size or ioff > size or isz != size - ioff: raise ValueError(f'incomplete or invalid RomFS image: {img}')
        h, rest = _FNV_OFFSET, b''
        f.seek(ioff)
        def mix(h, data):
            for w in struct.unpack(f'<{len(data) // 8}Q', data): h = ((h ^ w) * _FNV_PRIME) & _MASK
            return h
        h = mix(h, head)
        left = isz
        while left:
            chunk = f.read(min(left, 8 << 20))
            if not chunk: raise ValueError(f'incomplete RomFS image: {img}')
            left -= len(chunk)
            chunk = rest + chunk
            keep = len(chunk) % 8 if left else 0
            rest = chunk[len(chunk) - keep:]; chunk = chunk[:len(chunk) - keep]
            if not left and len(chunk) % 8: chunk += b'\0' * (8 - len(chunk) % 8)
            h = mix(h, chunk)
    return f'{ID_MAGIC} title={_clean(title)} version={_clean(version)} build={_clean(build)} size={size} index={h:016x}'


def parse_identity(text):   # {field: value} of an identity line; size is an int
    parts = text.split()
    if not parts or parts[0] != ID_MAGIC: raise ValueError('not an identity line')
    d = dict(p.split('=', 1) for p in parts[1:] if '=' in p)
    if 'size' not in d or 'index' not in d: raise ValueError('identity line without size or index')
    d['size'] = int(d['size']); return d


def same_identity(a, b):   # two lines name the same image when size and index agree (title, version and build only describe it)
    x, y = parse_identity(a), parse_identity(b)
    return (x['size'], x['index']) == (y['size'], y['index'])


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--identity']:   # romfs_image.py --identity IMAGE [TITLE [VERSION [BUILD]]]: the identity line of IMAGE
        try: print(identity(*sys.argv[2:6]))
        except (OSError, ValueError, TypeError) as e: sys.exit(f'[RED] {e}')
        sys.exit(0)
    ap = argparse.ArgumentParser(description='Pack or overlay an indexed RomFS image')
    ap.add_argument('--overlay', action='store_true', help='apply an overlay folder to an existing image')
    ap.add_argument('--reuse', action='store_true', help='keep the image only while its source tree, packer and image are unchanged')
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()
    if args.overlay:
        if len(args.paths) != 3: ap.error('--overlay requires BASE_IMAGE OVERLAY_FOLDER OUT_IMAGE')
        n = overlay_image(*args.paths, reuse=args.reuse); image_path = args.paths[2]
    else:
        if len(args.paths) != 2: ap.error('requires ROMFS_FOLDER OUT_IMAGE')
        n = pack(*args.paths, reuse=args.reuse); image_path = args.paths[1]
    print(f'{image_path}: {n} entries, {os.path.getsize(image_path)} bytes')
