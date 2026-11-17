"""Would a per-speaker mispronunciation glossary help? Aggregate-only check on APROCSA.

Collects participant (*PAR) paraphasias coded in CHAT as `produced [: target] [* code]`
and reports, per speaker:
  - counts by type (p phonological, s semantic, n neologism, o other);
  - a "pre-session" glossary: (produced, target) pairs from the first half of the
    transcript, and how many second-half paraphasias it would catch;
  - an "in-session" glossary that learns each pair as it occurs, and how many later
    paraphasias it would catch (same form and same target seen earlier).
Prints counts only, never transcript text (the data's terms forbid redistribution).

Usage: python -I -X utf8 scripts/paraphasia_consistency.py data/aprocsa/cha
"""
import collections
import glob
import os
import re
import sys

PAT = re.compile(r'(<[^>]+>|[^\s\[\]<>]+)\s*\[:\s*([^\]]+)\]\s*\[\*\s*([^\]]+)\]')


def norm(s):
    s = s.strip('<>').lower()
    s = re.sub(r"[^a-z' ]", '', s)
    return re.sub(r'\s+', ' ', s).strip()


def participant_lines(path):
    tiers, cur = [], None
    for raw in open(path, encoding='utf-8', errors='replace'):
        line = raw.rstrip('\n')
        if line.startswith('\t') and cur is not None:
            cur += ' ' + line.strip()
            continue
        if cur is not None:
            tiers.append(cur)
        cur = line if line[:1] in ('*', '%', '@') else None
    if cur is not None:
        tiers.append(cur)
    return [t[5:] for t in tiers if t.startswith('*PAR:')]


def events_for(path):
    out = []
    for line in participant_lines(path):
        for m in PAT.finditer(line):
            prod, tgt, code = norm(m.group(1)), norm(m.group(2)), m.group(3).strip().lower()
            if prod and tgt and prod != tgt:
                out.append((prod, tgt, code[0] if code[:1] in ('p', 's', 'n') else 'o'))
    return out


def main(cha_dir):
    tot = collections.Counter()
    print('speaker events p/s/n/o targets pre-session-caught in-session-caught')
    for path in sorted(glob.glob(os.path.join(cha_dir, '*.cha'))):
        ev = events_for(path)
        kinds = collections.Counter(e[2] for e in ev)
        pairs = [(e[0], e[1]) for e in ev]
        half = len(pairs) // 2
        first = set(pairs[:half])
        pre = sum(1 for p in pairs[half:] if p in first)
        online = sum(1 for k, p in enumerate(pairs) if p in set(pairs[:k]))
        spk = os.path.basename(path)[7:11]
        print('%s %d %d/%d/%d/%d %d %d of %d %d of %d' % (
            spk, len(ev), kinds['p'], kinds['s'], kinds['n'], kinds['o'],
            len({e[1] for e in ev}), pre, len(pairs) - half, online, len(pairs)))
        tot.update({'events': len(ev), 'p': kinds['p'], 's': kinds['s'], 'n': kinds['n'],
                    'o': kinds['o'], 'targets': len({e[1] for e in ev}), 'pre': pre,
                    'second': len(pairs) - half, 'online': online})
    print('ALL %(events)d %(p)d/%(s)d/%(n)d/%(o)d %(targets)d %(pre)d of %(second)d %(online)d of %(events)d' % tot)


if __name__ == '__main__':
    main(sys.argv[1])
