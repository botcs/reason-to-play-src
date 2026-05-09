# Translating Roomworld Layouts to VGDL

This document describes how to convert roomworld layout files (from the
roomworld-analysis dataset) into VGDL level files for the `roomworld_v0` game.

## Source Format

Roomworld layouts are 11x11 CSV grids. Each cell is comma-separated and
contains one of:

| Cell    | Meaning                                      |
|---------|----------------------------------------------|
| ` `     | Open floor                                   |
| `-`     | Wall                                         |
| `i`     | Avatar start (exactly one per layout)         |
| `x`     | Goal (exactly one per layout)                 |
| `k1`-`k6` | Key instances                              |
| `d1`-`d6` | Door instances (paired with matching key)  |
| `t1`-`t6` | Teleporter instances (paired or unpaired)  |
| `c1`-`c6` | Catapult instances (standalone)            |

Tool pairing uses numeric suffix: kN unlocks dN, tN teleports to the other tN.
Unpaired teleporters (only one instance of a given tN) are fake and do nothing.
Catapults never pair with anything.

## Target Format

VGDL level files are plain-text grids where each character maps to one or more
sprites via the LevelMapping in `roomworld.txt`. Every cell implicitly sits on
a floor tile.

## Grid Transformation

The 11x11 source grid becomes a 13x13 VGDL grid by adding a 1-tile-thick
outer wall border:

```
Row 0:  wwwwwwwwwwwww          <- added top wall
Row 1:  w<inner row 0 >w      <- source rows wrapped with 'w'
...
Row 11: w<inner row 10>w
Row 12: wwwwwwwwwwwww          <- added bottom wall
```

Source column N maps to VGDL column N+1. Source row N maps to VGDL row N+1.

## Cell-to-Character Mapping

| Source cell | VGDL char | Sprites            | Notes                                    |
|-------------|-----------|--------------------|------------------------------------------|
| ` ` (space) | `.`       | floor              |                                          |
| `-`         | `w`       | floor wall         |                                          |
| `i`         | `A`       | floor avatar       |                                          |
| `x`         | `x`       | floor goal         |                                          |
| `c1`-`c6`  | `c`       | floor catapult     | All catapult instances share one type     |
| `k1`        | `K`       | floor key1         | Pairs with d1                            |
| `d1`        | `D`       | floor door1        | Opened by key1                           |
| `k6`        | `k`       | floor key6         | Pairs with d6                            |
| `d6`        | `G`       | floor door6        | Opened by key6                           |
| `k4`        | `e`       | floor key4         | Pairs with d4 (no door4 sprite yet)      |
| paired `tN` | `T`       | floor t6 (Portal)  | Both instances use same char; teleportToOther handles pairing |
| 2nd paired `tN` | `P`   | floor tp4 (Portal) | Use when layout has 2+ distinct teleporter pairs |
| unpaired `tN`| `t`      | floor teleporter   | Fake teleporter, no effect               |

## Current Limitations

The game description (`roomworld.txt`) was built incrementally for specific
levels and does NOT yet cover all 6 key/door/teleporter instance slots. The
table above shows what is currently wired up. To support a layout that uses
an unmapped tool instance, you must:

1. Add a new sprite in the SpriteSet (e.g. `key2 > Resource ... limit=1`)
2. Add a LevelMapping character for it (e.g. `L > floor key2`)
3. Add InteractionSet rules (changeResource + killSprite for keys,
   stepBackIfHasLess for doors, teleportToOther for paired teleporters)
4. Add the EOS cleanup rule (`key2 EOS > killSprite`)

### Instance-to-Sprite Mapping Summary

Currently defined sprites and which roomworld instance IDs they cover:

| Roomworld instance | VGDL sprite  | VGDL char | Interaction              |
|--------------------|-------------|-----------|--------------------------|
| k1                 | key1        | K         | Collected as resource    |
| k4                 | key4        | e         | Collected as resource    |
| k6                 | key6        | k         | Collected as resource    |
| d1                 | door1       | D         | Opens with key1          |
| d2 (no key in level) | door2     | d         | Always blocks (stepBack) |
| d6                 | door6       | G         | Opens with key6          |
| 1st paired tN      | t6 (Portal) | T         | Bidirectional teleport   |
| 2nd paired tN      | tp4 (Portal)| P         | Bidirectional teleport (orange hexagon) |
| unpaired/fake tN   | teleporter  | t         | No effect                |
| fake (green)       | t3          | S         | No effect (visual only)  |
| c1-c6              | catapult    | c         | catapultForward          |

Keys k2, k3, k5 and doors d2-d5 have no dedicated sprites yet. When a layout
contains one of these, reuse an existing key/door sprite that is not otherwise
used in that level (e.g. map k5 to key1/`K` if no k1 appears in the same
layout). For doors without a matching key in the layout, use door2/`d`
(unconditional stepBack).

When a layout has multiple distinct paired teleporter types (e.g. both t2 and
t4 pairs), use `T` (t6) for one pair and `P` (tp4) for the other. Each Portal
sprite type teleports to its own sibling only. If a layout needs 3+ distinct
pairs, a new Portal sprite must be added to the game description.

## Worked Example

Source layout `r1_3_1.txt`:
```
 , ,k5,-, , , ,-, , ,
 , , ,-, , , ,-, , ,
 , , ,-, ,t2, ,-, , ,
-,-,-,-,-,-,-,-,-,-,-
 , , ,-, ,c5, ,-, , ,
 , , ,-, , , ,-, , ,i
 , , ,-, , , ,-, ,c6,
-,-,-,-,-,-,-,-,-,-,-
 , , ,-, , , ,-, , ,x
 , , ,-,c4, , ,-, , ,
 , , ,-, , , ,-, , ,
```

Tools present: k5 (key, no matching door), t2 (single instance = fake),
c4/c5/c6 (catapults).

Mapping decisions:
- k5 -> K (reuse key1, since no k1 in this layout)
- t2 -> t (fake teleporter)
- c4, c5, c6 -> c (catapult)

Result (`roomworld_lvl2.txt`):
```
wwwwwwwwwwwww
w..Kw...w...w
w...w...w...w
w...w.t.w...w
wwwwwwwwwwwww
w...w.c.w...w
w...w...w..Aw
w...w...w.c.w
wwwwwwwwwwwww
w...w...w..xw
w...wc..w...w
w...w...w...w
wwwwwwwwwwwww
```

## Worked Example 2: Multiple Tool Types

Source layout `test_1.txt`:
```
 , ,t4,-, , , ,-, , ,
 , , ,-, , , ,-,c2, ,
 , , ,-, , , ,-, , ,
-,-,-,-,-,-,-,-,-,-,d1
 ,c4, ,d6, , , ,-, , ,
t5, , ,-,t2, , ,-, , ,
 , , ,-,t3,i,k6,-,k1,t2,
-,-,-,-,-,-,-,-,-,-,-
 , , ,-, , , ,d2, , ,
 , , ,-, , , ,-, , ,x
k4, , ,-, , , ,-,t4, ,
```

Tools present: k1, k4, k6 (keys), d1, d2, d6 (doors), t2 (paired), t3 (fake),
t4 (paired), t5 (fake), c2/c4 (catapults).

Mapping decisions:
- k1 -> K (key1), k6 -> k (key6), k4 -> e (key4) -- direct matches
- d1 -> D (door1, opens with key1)
- d6 -> G (door6, opens with key6)
- d2 -> d (door2, always blocks -- no k2 in layout)
- t2 pair -> T (t6 Portal, purple hexagon)
- t4 pair -> P (tp4 Portal, orange hexagon)
- t3 -> S (fake teleporter, green hexagon)
- t5 -> t (fake teleporter, purple hexagon)
- c2, c4 -> c (catapult)

Result (`roomworld_lvl3.txt`):
```
wwwwwwwwwwwww
w..Pw...w...w
w...w...wc..w
w...w...w...w
wwwwwwwwwwwDw
w.c.G...w...w
wt..wT..w...w
w...wSAkwKT.w
wwwwwwwwwwwww
w...w...d...w
w...w...w..xw
we..w...wP..w
wwwwwwwwwwwww
```
