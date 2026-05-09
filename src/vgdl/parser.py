# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
from types import FunctionType

from src.vgdl import registry, ontology
from src.vgdl.core import BasicGame, SpriteRegistry
from src.vgdl.core import Effect, FunctionalEffect
from src.vgdl.ontology.terminations import MultiSpriteCounter, SpriteCounter
from src.vgdl.ontology.effects import killSprite

registry.register_all(ontology)
registry.register_class(BasicGame)


class Node(object):
    """Lightweight indented tree structure, with automatic insertion at the right spot."""

    parent = None

    def __init__(self, content, indent, parent=None):
        self.children = []
        self.content = content
        self.indent = indent
        if parent:
            parent.insert(self)
        else:
            self.parent = None

    def insert(self, node):
        if self.indent < node.indent:
            if len(self.children) > 0:
                assert self.children[0].indent == node.indent, (
                    "children indentations must match"
                )
            self.children.append(node)
            node.parent = self
        else:
            assert self.parent, "Root node too indented?"
            self.parent.insert(node)

    def __repr__(self):
        if len(self.children) == 0:
            return self.content
        else:
            return self.content + str(self.children)

    def get_root(self):
        if self.parent:
            return self.parent.get_root()
        else:
            return self


def indent_tree_parser(s, tabsize=8):
    """Produce an unordered tree from an indented string."""
    # insensitive to tabs, parentheses, commas
    s = s.expandtabs(tabsize)
    s.replace("(", " ")
    s.replace(")", " ")
    s.replace(",", " ")
    lines = s.split("\n")

    last = Node("", -1)
    for l in lines:
        # remove comments starting with "#"
        if "#" in l:
            l = l.split("#")[0]
        # handle whitespace and indentation
        content = l.strip()
        if len(content) > 0:
            indent = len(l) - len(l.lstrip())
            last = Node(content, indent, last)
    return last.get_root()


class VGDLParser:
    """
    Parses a string into
    """

    verbose = False

    def parse_game(self, tree, **kwargs):
        """Accepts either a string, or a tree."""

        if not isinstance(tree, Node):
            tree = indent_tree_parser(tree).children[0]
        sclass, args = self._parse_args(tree.content)
        args.update(kwargs)

        # Basic Game construction
        self.sprite_registry = SpriteRegistry()
        self.game = sclass(self.sprite_registry, **args)
        for c in tree.children:
            # _, args = self._parseArgs(' '.join(c.content.split(' ')[1:]))
            if c.content.startswith("SpriteSet"):
                self.parse_sprites(c.children)
            if c.content == "InteractionSet":
                self.parse_interactions(c.children)
            if c.content == "LevelMapping":
                self.parse_mappings(c.children)
            if c.content == "TerminationSet":
                self.parse_terminations(c.children)

        self.game.finish_setup()

        return self.game

    def parse_game_for_theory(self, tree):
        """
        Generate four lists storing
        Accepts either a string, or a tree.
        """
        if not isinstance(tree, Node):
            tree = indent_tree_parser(tree).children[0]
        sclass, args = self._parse_args(tree.content)

        sprites, interactions, mapping, terminations = [], [], [], []
        for c in tree.children:
            # _, args = self._parseArgs(' '.join(c.content.split(' ')[1:]))
            if c.content.startswith("SpriteSet"):
                sprites += self.parse_sprites_for_theory(c.children)
            if c.content == "InteractionSet":
                interactions += self.parse_interactions_for_theory(c.children)
            if c.content == "LevelMapping":
                mapping += self.parse_mappings_for_theory(c.children)
            if c.content == "TerminationSet":
                terminations += self.parse_terminations_for_theory(c.children)

        # unroll the tree of sprites
        parent_classes_dict = dict()
        old_to_new_names = dict(EOS="EOS")
        for (
            i_s,
            s,
        ) in enumerate(sprites):
            if len(s[-1]) > 1:
                parent_classes = s[-1][:-1]
                for p in parent_classes:
                    if p not in parent_classes_dict.keys():
                        parent_classes_dict[p] = []
                    parent_classes_dict[p].append(i_s)
                new_name = "--".join(s[-1])
                sprites[i_s] = (new_name, s[1], s[2], [new_name])
                old_to_new_names[s[0]] = new_name
            else:
                old_to_new_names[s[0]] = s[0]
        sprites_names = [s[0] for s in sprites]

        # unroll the interactions
        new_interactions = []
        for int in interactions:
            assert len(int[1]) == 2
            params = int[2]
            if "stype" in params.keys():
                params["stype"] = old_to_new_names[params["stype"]]
            if int[1][0] in parent_classes_dict.keys():
                new_ints = []
                for child_id in parent_classes_dict[int[1][0]]:
                    new_ints.append(
                        (
                            int[0],
                            [sprites_names[child_id], old_to_new_names[int[1][1]]],
                            params,
                        )
                    )
            elif int[1][1] in parent_classes_dict.keys():
                new_ints = []
                for child_id in parent_classes_dict[int[1][1]]:
                    new_ints.append(
                        (
                            int[0],
                            [old_to_new_names[int[1][0]], sprites_names[child_id]],
                            params,
                        )
                    )
            else:
                new_ints = [
                    (
                        int[0],
                        [old_to_new_names[int[1][0]], old_to_new_names[int[1][1]]],
                        params,
                    )
                ]
            for new_int in new_ints:
                if new_int[0].__name__ == "killBoth":
                    new_interactions.append((killSprite, new_int[1], new_int[2]))
                    new_interactions.append(
                        (killSprite, [new_int[1][1], new_int[1][0]], new_int[2])
                    )
                else:
                    new_interactions.append(new_int)

        # update mappings
        new_mapping = []
        for m in mapping:
            names, character = m
            new_names = [old_to_new_names.get(n, n) for n in names]
            new_mapping.append((new_names, character))

        # replace terminations with multi sprite counters when needed
        new_terminations = []
        for t in terminations:
            # track stypes
            stypes = []
            params = t[1]
            if t[0].__name__ not in ["SpriteCounter", "MultiSpriteCounter"]:
                new_terminations.append(t)
            else:
                if t[0].__name__ == "SpriteCounter":
                    if t[1]["stype"] in parent_classes_dict.keys():
                        for i, i_s in enumerate(parent_classes_dict[t[1]["stype"]]):
                            stypes.append(sprites_names[i_s])
                        del params["stype"]
                    else:
                        stypes.append(old_to_new_names[params["stype"]])
                elif t[0].__name__ == "MultiSpriteCounter":
                    stype_id = 0
                    while True:
                        stype_id += 1
                        if f"stype{stype_id}" in params.keys():
                            stype = params[f"stype{stype_id}"]
                            if stype in parent_classes_dict.keys():
                                for i, i_s in enumerate(parent_classes_dict[stype]):
                                    stypes.append(sprites_names[i_s])
                            else:
                                stypes.append(old_to_new_names[stype])
                        else:
                            break

                # now feed the stypes
                assert len(stypes) > 0
                if len(stypes) == 1:
                    params["stype"] = stypes[0]
                    new_terminations.append((SpriteCounter, params))
                else:
                    for i, stype in enumerate(stypes):
                        params[f"stype{i + 1}"] = stype
                    new_terminations.append((MultiSpriteCounter, params))

        return sprites, new_interactions, new_mapping, new_terminations, args

    def _eval(self, estr):
        """
        Whatever is visible in the global namespace (after importing the ontologies)
        can be used in the VGDL, and is evaluated.
        """
        # Classes and functions etc are registered with the global registry
        if estr in registry:
            return registry.request(estr)
        else:
            # Strings and numbers should just be interpreted
            return eval(estr)

    def parse_interactions(self, inodes):
        for inode in inodes:
            if ">" in inode.content:
                pair, edef = [x.strip() for x in inode.content.split(">")]
                eclass, kwargs = self._parse_args(edef)
                objs = [x.strip() for x in pair.split(" ") if len(x) > 0]

                # Create an effect for each actee
                for obj in objs[1:]:
                    args = [objs[0], obj]

                    if isinstance(eclass, FunctionType):
                        effect = FunctionalEffect(eclass, *args, **kwargs)
                    else:
                        assert issubclass(eclass, Effect)
                        effect = eclass(*args, **kwargs)

                    self.game.collision_eff.append(effect)

                if self.verbose:
                    print("Collision", pair, "has effect:", effect)

    def parse_interactions_for_theory(self, inodes):
        interactions = []
        for inode in inodes:
            if ">" in inode.content:
                pair, edef = [x.strip() for x in inode.content.split(">")]
                eclass, kwargs = self._parse_args(edef)
                objs = [x.strip() for x in pair.split(" ") if len(x) > 0]

                # Create an effect for each actee
                for obj in objs[1:]:
                    args = [objs[0], obj]
                    interactions.append((eclass, args, kwargs))
        return interactions

    def parse_terminations(self, tnodes):
        for tn in tnodes:
            sclass, args = self._parse_args(tn.content)
            if self.verbose:
                print("Adding:", sclass, args)
            self.game.terminations.append(sclass(**args))

    def parse_terminations_for_theory(self, tnodes):
        terminations = []
        for tn in tnodes:
            sclass, args = self._parse_args(tn.content)
            terminations.append((sclass, args))
        return terminations

    def parse_sprites_for_theory(
        self,
        snodes,
        parentclass=None,
        parentargs={},
        parenttypes=[],
        sprites=[],
        depth=0,
    ):
        if depth == 0:
            sprites = []
        for sn in snodes:
            assert ">" in sn.content
            key, sdef = [x.strip() for x in sn.content.split(">")]
            sclass, args = self._parse_args(sdef, parentclass, parentargs.copy())
            stypes = parenttypes + [key]
            if len(sn.children) == 0:
                sprites += [(key, sclass, args, stypes)]
            else:
                sprites = self.parse_sprites_for_theory(
                    sn.children, sclass, args, stypes, sprites=sprites, depth=depth + 1
                )
        return sprites

    def parse_sprites(self, snodes, parentclass=None, parentargs={}, parenttypes=[]):
        for sn in snodes:
            assert ">" in sn.content
            key, sdef = [x.strip() for x in sn.content.split(">")]
            sclass, args = self._parse_args(sdef, parentclass, parentargs.copy())
            stypes = parenttypes + [key]

            if "singleton" in args:
                if args["singleton"] == True:
                    self.sprite_registry.register_singleton(key)
                args = args.copy()
                del args["singleton"]

            if len(sn.children) == 0:
                if self.verbose:
                    print("Defining:", key, sclass, args, stypes)
                self.sprite_registry.register_sprite_class(key, sclass, args, stypes)
                if key in self.game.sprite_order:
                    # last one counts
                    self.game.sprite_order.remove(key)
                self.game.sprite_order.append(key)
            else:
                self.parse_sprites(sn.children, sclass, args, stypes)

    def parse_mappings(self, mnodes):
        for mn in mnodes:
            c, val = [x.strip() for x in mn.content.split(">")]
            assert len(c) == 1, "Only single character mappings allowed."
            # a char can map to multiple sprites
            keys = [x.strip() for x in val.split(" ") if len(x) > 0]
            if self.verbose:
                print("Mapping", c, keys)
            self.game.char_mapping[c] = keys

    def parse_mappings_for_theory(self, mnodes):
        mappings = []
        for mn in mnodes:
            c, val = [x.strip() for x in mn.content.split(">")]
            assert len(c) == 1, "Only single character mappings allowed."
            # a char can map to multiple sprites
            keys = [x.strip() for x in val.split(" ") if len(x) > 0]
            mappings.append((keys, c))
        return mappings

    def _parse_args(self, s, sclass=None, args=None):
        if not args:
            args = {}
        sparts = [x.strip() for x in s.split(" ") if len(x) > 0]
        if len(sparts) == 0:
            return sclass, args
        if "=" not in sparts[0]:
            sclass = self._eval(sparts[0])
            sparts = sparts[1:]
        for sp in sparts:
            try:
                k, val = sp.split("=")
            except:
                stop = 1
            try:
                args[k] = self._eval(val)
            except:
                args[k] = val
        return sclass, args
