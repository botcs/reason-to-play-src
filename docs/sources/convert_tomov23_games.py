#!/usr/bin/env python3
"""
Convert VGDL games from tomov23-neuron-reference format to infer-vgdl format.

This script converts game files from the tomov23 VGDL grammar to the current
infer-vgdl grammar, handling syntax differences and ensuring compatibility.
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def scan_source_games(source_dir: Path) -> dict:
    """
    Scan source directory and organize files by game name and version.

    Args:
        source_dir: Path to directory containing vgfmri*.txt files

    Returns:
        dict mapping (game_name, version) to dict with 'main_file' and 'level_files'
        Example: {('bait', 'vgfmri3'): {'main_file': Path(...), 'level_files': [(0, Path(...)), ...]}}
    """
    games = defaultdict(lambda: {'main_file': None, 'level_files': []})

    for file_path in source_dir.glob("*.txt"):
        filename = file_path.name

        # Parse filename: vgfmriN_gamename[_lvlM].txt
        match = re.match(r'(vgfmri[34])_(.+?)(?:_lvl(\d+))?\.txt$', filename)
        if not match:
            continue

        version = match.group(1)  # vgfmri3 or vgfmri4
        game_name = match.group(2)
        level_num = match.group(3)

        key = (game_name, version)

        if level_num is not None:
            # Level file
            games[key]['level_files'].append((int(level_num), file_path))
        else:
            # Main game file
            games[key]['main_file'] = file_path

    # Sort level files by level number
    for game_info in games.values():
        game_info['level_files'].sort(key=lambda x: x[0])

    return dict(games)


def parse_sections(content: str) -> dict:
    """
    Parse VGDL content into sections, handling any input ordering.

    Args:
        content: Full VGDL script as string

    Returns:
        dict with keys: header, spriteset, levelmapping, interactionset, terminationset
        Each value is a list of lines (without the section header line)
    """
    # Normalize indentation: convert tabs to spaces (8 spaces per tab for VGDL)
    content = content.replace('\t', '        ')

    lines = content.split('\n')

    sections = {
        'header': [],
        'spriteset': [],
        'levelmapping': [],
        'interactionset': [],
        'terminationset': []
    }

    current_section = 'header'

    for line in lines:
        stripped = line.strip()

        # Detect section headers (case-sensitive)
        if stripped == 'SpriteSet':
            current_section = 'spriteset'
            continue
        elif stripped == 'LevelMapping':
            current_section = 'levelmapping'
            continue
        elif stripped == 'InteractionSet':
            current_section = 'interactionset'
            continue
        elif stripped == 'TerminationSet':
            current_section = 'terminationset'
            continue

        # Add line to current section
        sections[current_section].append(line)

    return sections


def extract_sprite_names(spriteset_lines: list) -> list:
    """
    Extract all sprite names from SpriteSet section, including hierarchical children.

    Args:
        spriteset_lines: Lines from SpriteSet section

    Returns:
        List of sprite names in order of appearance
    """
    sprite_names = []

    for line in spriteset_lines:
        if '>' in line:
            # Match sprite name before '>'
            match = re.match(r'\s*(\w+)\s*>', line)
            if match:
                sprite_names.append(match.group(1))

    return sprite_names


def transform_spriteset(lines: list) -> tuple:
    """
    Transform SpriteSet section: add floor, convert color to img, handle hierarchy.

    Flattens hierarchical sprite definitions by:
    - Removing parent-only definitions (e.g., "citizen >")
    - Promoting child sprites to top level
    - Tracking removed parents for reference remapping

    Args:
        lines: Lines from SpriteSet section

    Returns:
        Tuple of (transformed_lines, sprite_names, parent_to_children_map)
    """
    # First pass: identify parent-only definitions and mark lines to skip/adjust
    parent_indices = {}  # Maps line index to parent info
    parent_to_children = {}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith('>') and not stripped.endswith('>>') and '=' not in stripped:
            # This is a parent-only definition
            parent_name = stripped.rstrip('>').strip()
            indent_match = re.match(r'(\s*)', line)
            parent_indent = len(indent_match.group(1)) if indent_match else 0
            parent_indices[i] = {
                'name': parent_name,
                'indent': parent_indent
            }
            parent_to_children[parent_name] = []

    # Second pass: process lines, skip parents, adjust children indentation
    result = []
    sprite_names = []
    has_floor = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            result.append(line)
            i += 1
            continue

        # Skip parent-only definitions
        if i in parent_indices:
            parent_info = parent_indices[i]
            i += 1
            continue

        # Get current indentation
        indent_match = re.match(r'(\s*)', line)
        current_indent = len(indent_match.group(1)) if indent_match else 0

        # Check if this is a child of a recently removed parent
        # by finding the nearest parent before this line
        parent_info = None
        for j in range(i - 1, -1, -1):
            if j in parent_indices:
                candidate = parent_indices[j]
                # Check if current line is indented more than this parent
                if current_indent > candidate['indent']:
                    parent_info = candidate
                    break

        # If this is a child of a removed parent, reduce indentation
        if parent_info:
            # Reduce indentation to parent's level
            line = ' ' * parent_info['indent'] + line.lstrip()

            # Track as child
            child_match = re.match(r'\s*(\w+)\s*>', line)
            if child_match:
                parent_to_children[parent_info['name']].append(child_match.group(1))

        # Extract sprite name
        match = re.match(r'\s*(\w+)\s*>', line)
        if match:
            sprite_name = match.group(1)
            sprite_names.append(sprite_name)
            if sprite_name == 'floor':
                has_floor = True

        # Normalize indentation: ensure all sprite definitions use 8 spaces
        # (or multiples of 4 for hierarchical sprites)
        indent_match = re.match(r'(\s*)', line)
        current_indent = len(indent_match.group(1)) if indent_match else 0
        if current_indent > 0 and current_indent < 8:
            # Normalize to 8 spaces for top-level sprites
            line = '        ' + line.lstrip()
        elif current_indent >= 8:
            # Normalize to nearest multiple of 4
            normalized_indent = ((current_indent + 3) // 4) * 4
            line = ' ' * normalized_indent + line.lstrip()

        # Convert FIRST color= to img=colors/
        transformed = re.sub(
            r'\bcolor=(\w+)',
            r'img=colors/\1',
            line,
            count=1
        )
        result.append(transformed)

        i += 1

    # Add floor sprite if missing
    if not has_floor:
        result.insert(0, '        floor > Immovable img=colors/LIGHTGRAY')
        sprite_names.insert(0, 'floor')

    return result, sprite_names, parent_to_children


def transform_levelmapping(lines: list, sprite_names: list) -> list:
    """
    Transform LevelMapping section: add floor prefix to all mappings.

    Also auto-adds default VGDL mappings when sprites are defined but their
    standard character mappings are missing (common bug in tomov23 source files).

    Args:
        lines: Lines from LevelMapping section
        sprite_names: List of defined sprite names (to filter undefined references)

    Returns:
        Transformed lines with floor prefixes
    """
    result = ['        . > floor']  # Add floor mapping first

    # Sprites that are commonly undefined or problematic in source files
    # (bugs in original tomov23 files that would also fail in original interpreter)
    undefined_sprites = {'rand', 'missile1', 'missile2', 'missile', 'mover'}

    # Track which default mappings we find
    # VGDL convention: 'A' -> avatar, 'w' -> wall
    found_mappings = set()

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Check if line contains mapping
        if '>' in stripped:
            parts = stripped.split('>', 1)
            if len(parts) == 2:
                char_part = parts[0].strip()
                sprite_part = parts[1].strip()

                # Track the character being mapped
                found_mappings.add(char_part)

                # Check if any referenced sprite is in the known undefined list
                referenced_sprites = sprite_part.split()
                if any(s in undefined_sprites for s in referenced_sprites):
                    # Skip this mapping - references known undefined sprite
                    continue

                # Add floor prefix with proper indentation
                new_line = f"        {char_part} > floor {sprite_part}"
                result.append(new_line)

    # Auto-add default VGDL mappings when sprites are defined but mappings are missing
    # This fixes bugs in original tomov23 source files that relied on implicit defaults
    default_mappings = [
        ('A', 'avatar'),
        ('w', 'wall'),
    ]

    for char, sprite in default_mappings:
        if sprite in sprite_names and char not in found_mappings:
            result.insert(1, f'        {char} > floor {sprite}')

    return result


def transform_interactionset(lines: list, sprite_names: list) -> list:
    """
    Transform InteractionSet section: fix spacing, merge changeScore, add EOS.

    Handles three interpreter differences between tomov23 and infer-vgdl:

    1. changeScore merge: The tomov23 format uses standalone "changeScore value=N"
       lines. The VGDL framework already handles scoreChange as a universal parameter
       on all effects via the Effect base class, so we merge the score onto the
       companion effect for the same sprite pair.

    2. transformTo execution order: The tomov23 interpreter sorts effects by priority,
       while infer-vgdl processes them in declaration order with lazy-cached sprite
       lists. When "A B > killSprite" and "B A > transformTo stype=C" coexist, B may
       already be C by the time the kill is evaluated, so we rewrite to "A C > killSprite".

    3. stepBack conflicts: Removes conflicting stepBack interactions that would prevent
       kill conditions from triggering.

    NOTE: plaqueAttack has manually adjusted scoreChange values (e.g., -3 in source
    vs -1 in repo) that represent deliberate game-balance changes. These cannot be
    handled by this converter and require manual post-conversion adjustment.

    Args:
        lines: Lines from InteractionSet section
        sprite_names: All sprite names for EOS interactions

    Returns:
        Transformed lines with EOS interactions added
    """
    result = []

    # Sprites that are commonly undefined or problematic in source files
    # (bugs in original tomov23 files that would also fail in original interpreter)
    undefined_sprites = {'rand', 'missile1', 'missile2', 'missile', 'mover'}

    # First pass: collect changeScore, transformTo targets, and stepBack conflicts
    change_scores = {}  # Maps (spriteA, spriteB) -> score value string
    transform_targets = {}  # Maps (B, A) -> C for "B A > transformTo stype=C"
    sprites_that_kill_avatar_conditionally = set()

    for line in lines:
        if '>' not in line:
            continue
        stripped = line.strip()

        # Collect changeScore entries: "A B > changeScore value=N"
        cs_match = re.match(r'(\w+)\s+(\w+)\s*>\s*changeScore\s+value=(-?\d+)', stripped)
        if cs_match:
            pair = (cs_match.group(1), cs_match.group(2))
            change_scores[pair] = cs_match.group(3)
            continue

        # Collect transformTo targets: "B A > transformTo stype=C"
        tt_match = re.match(r'(\w+)\s+(\w+)\s*>\s*transformTo\s+stype=(\w+)', stripped)
        if tt_match:
            pair = (tt_match.group(1), tt_match.group(2))
            transform_targets[pair] = tt_match.group(3)

        # Find sprites where "X avatar > killIfOtherHasMore" or "X avatar > killSprite"
        match = re.match(r'(\w+)\s+avatar\s*>\s*(killIfOtherHasMore|killSprite)', stripped)
        if match:
            sprites_that_kill_avatar_conditionally.add(match.group(1))

    # Second pass: process lines, applying all transformations
    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        # Skip changeScore lines entirely (already collected in first pass)
        stripped = line.strip()
        if re.match(r'\w+\s+\w+\s*>\s*changeScore', stripped):
            continue

        # Handle "> nothing" interactions: sword projectiles should be destroyed
        # on impact, so convert "sword X > nothing" to "sword X > killSprite".
        # All other "> nothing" interactions are skipped (not supported).
        if '> nothing' in line:
            stripped_check = line.strip()
            if stripped_check.startswith('sword '):
                line = line.replace('> nothing', '> killSprite')
            else:
                continue

        # Check if interaction references known undefined sprites
        if '>' in line:
            pair_part = line.split('>')[0].strip()
            sprites_in_interaction = [s.strip() for s in pair_part.split() if s.strip()]
            # Skip if any sprite in the interaction is in the known undefined list
            if any(s in undefined_sprites for s in sprites_in_interaction):
                continue

        # Skip "avatar X > stepBack" when X has a kill interaction with avatar
        # (stepBack prevents avatar from reaching X, breaking the kill condition)
        stripped = line.strip()
        stepback_match = re.match(r'avatar\s+(\w+)\s*>\s*stepBack', stripped)
        if stepback_match:
            target_sprite = stepback_match.group(1)
            if target_sprite in sprites_that_kill_avatar_conditionally:
                continue

        # Fix spacing: >effect -> > effect
        line = re.sub(r'>([a-zA-Z])', r'> \1', line)
        stripped = line.strip()

        # Fix transformTo execution order: "A B > killSprite" where
        # "B A > transformTo stype=C" exists -> rewrite to "A C > killSprite"
        kill_match = re.match(r'(\w+)\s+(\w+)\s*>\s*killSprite(.*)', stripped)
        if kill_match:
            a, b, rest = kill_match.group(1), kill_match.group(2), kill_match.group(3)
            reverse_pair = (b, a)
            if reverse_pair in transform_targets:
                c = transform_targets[reverse_pair]
                stripped = f'{a} {c} > killSprite{rest}'

        # Merge scoreChange from changeScore entries onto companion effects
        pair_match = re.match(r'(\w+)\s+(\w+)\s*>', stripped)
        if pair_match:
            pair = (pair_match.group(1), pair_match.group(2))
            if pair in change_scores:
                score_val = change_scores.pop(pair)
                stripped = stripped + f' scoreChange={score_val}'

        # Normalize indentation to 8 spaces for all non-empty lines
        if stripped:
            line = '        ' + stripped

        result.append(line)

    # All changeScore entries must have been merged onto a companion effect
    if change_scores:
        unmerged = ', '.join(f'{a} {b} (value={v})' for (a, b), v in change_scores.items())
        raise ValueError(
            f"changeScore entries have no companion effect to merge onto: {unmerged}"
        )

    # Add EOS interactions for all sprites
    result.append('')  # Blank line before EOS section
    for sprite in sprite_names:
        result.append(f'        {sprite} EOS > killSprite')

    return result


def remap_parent_references(lines: list, parent_to_children: dict) -> list:
    """
    Remap references to removed parent sprites (e.g., stype=parent -> stype=child).

    Args:
        lines: Lines that may contain parent references
        parent_to_children: Mapping of removed parent names to their children

    Returns:
        Lines with remapped references
    """
    result = []

    for line in lines:
        # Find and replace stype=parent with stype=child
        for parent, children in parent_to_children.items():
            if children:  # If parent had children
                # Use the first child as the replacement
                child = children[0]
                # Replace stype=parent with stype=child
                line = re.sub(
                    rf'\bstype={parent}\b',
                    f'stype={child}',
                    line
                )

        result.append(line)

    return result


def transform_terminationset(lines: list) -> list:
    """
    Transform TerminationSet section: remove unsupported bonus parameter.

    Args:
        lines: Lines from TerminationSet section

    Returns:
        Transformed lines
    """
    result = []

    for line in lines:
        # Remove bonus= parameter (not supported in current parser)
        line = re.sub(r'\s+bonus=-?[\d.]+', '', line)

        # Normalize indentation to 8 spaces for all non-empty lines
        stripped = line.strip()
        if stripped:
            line = '        ' + stripped

        result.append(line)

    return result


def convert_vgdl_syntax(content: str) -> str:
    """
    Convert VGDL syntax from tomov23 format to infer-vgdl format.

    This is the main orchestrator function that applies all transformations.

    Args:
        content: Original VGDL script

    Returns:
        Converted VGDL script
    """
    # 1. Parse into sections
    sections = parse_sections(content)

    # 2. Transform SpriteSet and extract sprite names
    spriteset_lines, sprite_names, parent_to_children = transform_spriteset(sections['spriteset'])

    # 3. Remap parent references in SpriteSet (for stype= parameters)
    spriteset_lines = remap_parent_references(spriteset_lines, parent_to_children)

    # 4. Transform LevelMapping (pass sprite_names to filter undefined references)
    levelmapping_lines = transform_levelmapping(sections['levelmapping'], sprite_names)

    # 5. Transform InteractionSet with EOS
    interactionset_lines = transform_interactionset(sections['interactionset'], sprite_names)

    # 6. Transform TerminationSet (remove bonus parameters)
    terminationset_lines = transform_terminationset(sections['terminationset'])

    # 7. Reassemble in correct order
    result_lines = []

    # Header (BasicGame line)
    for line in sections['header']:
        result_lines.append(line)

    # SpriteSet
    result_lines.append('    SpriteSet')
    result_lines.extend(spriteset_lines)

    # LevelMapping
    result_lines.append('')
    result_lines.append('    LevelMapping')
    result_lines.extend(levelmapping_lines)

    # InteractionSet
    result_lines.append('')
    result_lines.append('    InteractionSet')
    result_lines.extend(interactionset_lines)

    # TerminationSet
    result_lines.append('')
    result_lines.append('    TerminationSet')
    result_lines.extend(terminationset_lines)

    return '\n'.join(result_lines)


def to_camel_case(snake_case: str) -> str:
    """
    Convert snake_case or lowercase to lowerCamelCase.

    Args:
        snake_case: Name in snake_case or lowercase

    Returns:
        Name in lowerCamelCase

    Examples:
        avoidgeorge -> avoidGeorge
        plaque_attack -> plaqueAttack
        bait -> bait
    """
    # Split by underscores
    parts = snake_case.split('_')

    if len(parts) == 1:
        # No underscores - capitalize subsequent characters at word boundaries
        # For simple cases like "avoidgeorge", we need to detect word boundaries
        # For now, just capitalize after known prefixes
        name = parts[0]

        # Handle common patterns
        if name.startswith('avoid'):
            return 'avoid' + name[5:].capitalize()
        elif name.startswith('plaque'):
            return 'plaque' + name[6:].capitalize()
        else:
            return name
    else:
        # Has underscores - standard camelCase conversion
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])


def convert_game(game_key: tuple, game_info: dict, target_dir: Path,
                dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Convert a single game with all its files.

    Args:
        game_key: Tuple of (game_name, version)
        game_info: Dict with 'main_file' and 'level_files'
        target_dir: Target directory for converted games
        dry_run: If True, don't actually write files
        verbose: If True, print detailed information

    Returns:
        True if successful, False otherwise
    """
    game_name, version = game_key
    main_file = game_info['main_file']
    level_files = game_info['level_files']

    if not main_file:
        print(f"WARNING: No main file found for game '{game_name}' ({version}), skipping...")
        return False

    # Convert to camelCase and create target directory name
    camel_name = to_camel_case(game_name)
    # Base name for files includes version (e.g., plaqueAttack_vgfmri3)
    base_name = f"{camel_name}_{version}"
    target_game_dir = target_dir / f"{base_name}_v0"

    if dry_run:
        print(f"\nWould create: {target_game_dir}")
        print(f"  Main file: {main_file.name} -> {base_name}.txt")
        print(f"  Levels: {len(level_files)} files")
        return True

    # Create target directory
    target_game_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\nProcessing: {base_name}_v0")

    # Read and convert main game file
    with open(main_file, 'r') as f:
        content = f.read()

    converted_content = convert_vgdl_syntax(content)

    # Write converted main file (must match directory basename without _v0)
    target_main_file = target_game_dir / f"{base_name}.txt"
    with open(target_main_file, 'w') as f:
        f.write(converted_content)

    if verbose:
        print(f"  Wrote main file: {target_main_file}")

    # Copy level files (no conversion needed)
    for level_num, level_file in level_files:
        target_level_file = target_game_dir / f"{base_name}_lvl{level_num}.txt"
        shutil.copy2(level_file, target_level_file)

    if verbose:
        print(f"  Copied {len(level_files)} level files")

    return True


def validate_game(game_dir: Path, verbose: bool = False) -> bool:
    """
    Validate a converted game by loading it with VGDLParser.

    Args:
        game_dir: Path to converted game directory
        verbose: If True, print detailed validation info

    Returns:
        True if valid, False otherwise
    """
    # Import here to avoid dependency if --validate not used
    sys.path.append('/home/ubuntu/infer-vgdl')
    import src.vgdl as vgdl

    game_name = game_dir.name.replace('_vgfmri3_v0', '').replace('_vgfmri4_v0', '').replace('_v0', '')

    # Find main file
    rule_files = [f for f in game_dir.glob('*.txt') if '_lvl' not in f.name]
    if not rule_files:
        print(f"ERROR: No main file found in {game_dir}")
        return False

    rule_file = rule_files[0]

    # Load and parse VGDL
    with open(rule_file, 'r') as f:
        vgdl_script = f.read()

    # Test parsing
    sprites, interactions, mapping, terminations, args = \
        vgdl.VGDLParser().parse_game_for_theory(vgdl_script)

    # Verify floor exists
    floor_found = any(s[0] == 'floor' for s in sprites)
    if not floor_found:
        print(f"FAILED: {game_name} missing floor sprite")
        return False

    # Verify all sprites use img= not color=
    for sprite in sprites:
        if 'color' in sprite[2] and sprite[0] != 'floor':
            print(f"FAILED: {game_name} sprite {sprite[0]} still uses color=")
            return False

    # Verify EOS interactions exist
    has_eos = any('EOS' in str(i) for i in interactions)
    if not has_eos:
        print(f"FAILED: {game_name} missing EOS interactions")
        return False

    if verbose:
        print(f"PASSED: {game_name}")

    return True


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert VGDL games from tomov23 format to infer-vgdl format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be converted
  ./convert_tomov23_games.py --dry-run

  # Convert all games
  ./convert_tomov23_games.py

  # Convert only vgfmri4 versions
  ./convert_tomov23_games.py --version vgfmri4

  # Convert specific games
  ./convert_tomov23_games.py --games avoidgeorge plaqueAttack

  # Convert with validation
  ./convert_tomov23_games.py --validate
        """
    )

    parser.add_argument('--source', type=str,
                       default='/home/ubuntu/infer-vgdl/tomov23-neuron-reference/all_games',
                       help='Source directory with tomov23 games')
    parser.add_argument('--target', type=str,
                       default='/home/ubuntu/infer-vgdl/games',
                       help='Target directory for converted games')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be converted without actually converting')
    parser.add_argument('--games', nargs='+', metavar='GAME',
                       help='Specific games to convert (default: all)')
    parser.add_argument('--version', choices=['vgfmri3', 'vgfmri4'],
                       help='Only convert games from specific version')
    parser.add_argument('--validate', action='store_true',
                       help='Validate converted games by loading them')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed conversion information')

    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    # Validate directories
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return 1

    if not target_dir.exists():
        print(f"ERROR: Target directory not found: {target_dir}")
        return 1

    # Scan source directory
    print(f"Scanning source directory: {source_dir}")
    games = scan_source_games(source_dir)

    print(f"\nFound {len(games)} game-version combinations:")
    for (game_name, version), info in sorted(games.items()):
        n_levels = len(info['level_files'])
        has_main = "YES" if info['main_file'] else "NO"
        print(f"  {game_name} ({version}): main={has_main}, {n_levels} levels")

    # Filter by specific games if requested
    if args.games:
        games = {k: v for k, v in games.items() if k[0] in args.games}
        print(f"\nFiltered to games: {args.games}")

    # Filter by version if requested
    if args.version:
        games = {k: v for k, v in games.items() if k[1] == args.version}
        print(f"\nFiltered to version: {args.version}")

    print(f"\nWill process {len(games)} game-version combinations")

    # Convert games
    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Converting games...")
    successful = 0
    failed = 0

    for game_key, info in sorted(games.items()):
        if convert_game(game_key, info, target_dir, dry_run=args.dry_run, verbose=args.verbose):
            successful += 1
        else:
            failed += 1

    if args.dry_run:
        print("\nDry run complete. Run without --dry-run to actually convert files.")
    else:
        print(f"\nConversion complete: {successful} successful, {failed} failed")

        # Validate if requested
        if args.validate:
            print("\nValidating converted games...")
            validated = 0
            validation_failed = 0

            for (game_name, version), info in sorted(games.items()):
                camel_name = to_camel_case(game_name)
                base_name = f"{camel_name}_{version}"
                game_dir = target_dir / f"{base_name}_v0"

                if game_dir.exists():
                    if validate_game(game_dir, verbose=args.verbose):
                        validated += 1
                    else:
                        validation_failed += 1

            print(f"\nValidation complete: {validated} passed, {validation_failed} failed")

    return 0


if __name__ == '__main__':
    sys.exit(main())
