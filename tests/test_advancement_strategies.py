"""
Tests for level advancement strategies (fixed_budget vs blocked_curricula).

Uses real game action sequences on bait_vgfmri4 levels 0 and 1 to produce
deterministic outcomes, then verifies the advancement logic.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.lrm.gameplay.agent import GameplayAgent
from agents.lrm.config import (
    BlockedCurriculaAdvancement,
    Config,
    FixedBudgetAdvancement,
    GameConfig,
    HarnessConfig,
    LLMConfig,
    LoggingConfig,
    parse_advancement,
    validate_config,
)


# ---------------------------------------------------------------------------
# Level 0 of bait_vgfmri4 reference layout
# ---------------------------------------------------------------------------
# Avatar (8, 1)  |  Key (15, 3)  |  Goal (3, 9)
#
# Winning path (27 steps, idle_frames=0 -> 27 frames):
#   RIGHT x7  (8,1) -> (15,1)
#   DOWN  x2  (15,1) -> (15,3)  -- collects key, +5
#   LEFT  x12 (15,3) -> (3,3)
#   DOWN  x6  (3,3) -> (3,9)    -- reaches goal with key -> WIN
#
# Death:
#   'reset' kills the avatar immediately (1 step, 1 frame).
# ---------------------------------------------------------------------------

WIN_L0: list[str] = (
    ["right"] * 7
    + ["down"] * 2  # collect key
    + ["left"] * 12
    + ["down"] * 6  # reach goal -> WIN
)
assert len(WIN_L0) == 27

# ---------------------------------------------------------------------------
# Level 1 of bait_vgfmri4 reference layout
# ---------------------------------------------------------------------------
# Avatar (4, 1)  |  Mushroom (8, 1)  |  Goal (16, 3)
# Mushroom (3, 8)  |  Key (13, 9)
#
# Winning path (26 steps, idle_frames=0 -> 26 frames):
#   DOWN  x8   (4,1) -> (4,9)
#   RIGHT x9   (4,9) -> (13,9)  -- collects key
#   UP    x6   (13,9) -> (13,3)
#   RIGHT x3   (13,3) -> (16,3) -- reaches goal with key -> WIN
# ---------------------------------------------------------------------------

WIN_L1: list[str] = (
    ["down"] * 8
    + ["right"] * 9  # collect key
    + ["up"] * 6
    + ["right"] * 3  # reach goal -> WIN
)
assert len(WIN_L1) == 26

RESET: list[str] = ["reset"]


def _make_fixed_budget_agent(
    tmp_dir: str,
    level_frame_budget: int,
    max_levels: int = 1,
) -> GameplayAgent:
    """Create a GameplayAgent with fixed_budget advancement.

    idle_frames=0 so 1 step = 1 frame, making budget arithmetic trivial.
    """
    cfg = Config(
        harness=HarnessConfig(
            rationale_mode="prompted-rationale",
            suggestion_level="elaborate",
        ),
        llm=LLMConfig(backend="mock", model=""),
        seed=0,
        game=GameConfig(
            game="bait_vgfmri4",
            start_level=0,
            max_levels=max_levels,
            idle_frames=0,
            advancement={
                "strategy": "fixed_budget",
                "level_frame_budget": level_frame_budget,
            },
        ),
        logging=LoggingConfig(output_dir=tmp_dir, wandb_project="", verbose=False),
    )
    validate_config(cfg)
    return GameplayAgent(cfg)


def _make_blocked_agent(
    tmp_dir: str,
    total_frame_budget: int,
    consecutive_wins_required: int = 2,
    max_levels: int = 9,
    max_consecutive_losses: int = 0,
) -> GameplayAgent:
    """Create a GameplayAgent with blocked_curricula advancement.

    idle_frames=0 so 1 step = 1 frame, making budget arithmetic trivial.
    """
    cfg = Config(
        harness=HarnessConfig(
            rationale_mode="prompted-rationale",
            suggestion_level="elaborate",
        ),
        llm=LLMConfig(backend="mock", model=""),
        seed=0,
        game=GameConfig(
            game="bait_vgfmri4",
            start_level=0,
            max_levels=max_levels,
            idle_frames=0,
            advancement={
                "strategy": "blocked_curricula",
                "total_frame_budget": total_frame_budget,
                "consecutive_wins_required": consecutive_wins_required,
                "max_consecutive_losses": max_consecutive_losses,
            },
        ),
        logging=LoggingConfig(output_dir=tmp_dir, wandb_project="", verbose=False),
    )
    validate_config(cfg)
    return GameplayAgent(cfg)


# =========================================================================
# Config parsing and validation
# =========================================================================


class TestAdvancementValidation:
    """Verify parse_advancement() and validate_config() strictness."""

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            parse_advancement({"strategy": "bogus", "level_frame_budget": 100})

    def test_missing_strategy_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            parse_advancement({"level_frame_budget": 100})

    def test_extra_key_on_fixed_budget_raises(self):
        with pytest.raises(ValueError, match="extra keys"):
            parse_advancement(
                {
                    "strategy": "fixed_budget",
                    "level_frame_budget": 100,
                    "consecutive_wins_required": 2,
                }
            )

    def test_extra_key_on_blocked_curricula_raises(self):
        with pytest.raises(ValueError, match="extra keys"):
            parse_advancement(
                {
                    "strategy": "blocked_curricula",
                    "total_frame_budget": 100,
                    "consecutive_wins_required": 2,
                    "level_frame_budget": 1200,
                }
            )

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_advancement({"strategy": "fixed_budget", "level_frame_budget": 0})

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_advancement(
                {"strategy": "blocked_curricula", "total_frame_budget": -1}
            )

    def test_zero_consecutive_wins_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_advancement(
                {
                    "strategy": "blocked_curricula",
                    "total_frame_budget": 100,
                    "consecutive_wins_required": 0,
                }
            )

    def test_valid_fixed_budget(self):
        adv = parse_advancement({"strategy": "fixed_budget", "level_frame_budget": 500})
        assert isinstance(adv, FixedBudgetAdvancement)
        assert adv.level_frame_budget == 500

    def test_valid_blocked_curricula(self):
        adv = parse_advancement(
            {
                "strategy": "blocked_curricula",
                "total_frame_budget": 2700,
                "consecutive_wins_required": 3,
            }
        )
        assert isinstance(adv, BlockedCurriculaAdvancement)
        assert adv.total_frame_budget == 2700
        assert adv.consecutive_wins_required == 3

    def test_validate_config_replaces_dict_with_dataclass(self):
        cfg = Config(
            game=GameConfig(
                advancement={
                    "strategy": "blocked_curricula",
                    "total_frame_budget": 100,
                    "consecutive_wins_required": 2,
                }
            )
        )
        validate_config(cfg)
        assert isinstance(cfg.game.advancement, BlockedCurriculaAdvancement)

    def test_defaults_applied_when_keys_omitted(self):
        adv = parse_advancement({"strategy": "fixed_budget"})
        assert adv.level_frame_budget == 1200

        adv = parse_advancement({"strategy": "blocked_curricula"})
        assert adv.total_frame_budget == 2700
        assert adv.consecutive_wins_required == 2
        assert adv.max_consecutive_losses == 0

    def test_max_consecutive_losses_parsed(self):
        adv = parse_advancement(
            {
                "strategy": "blocked_curricula",
                "total_frame_budget": 100,
                "consecutive_wins_required": 2,
                "max_consecutive_losses": 3,
            }
        )
        assert isinstance(adv, BlockedCurriculaAdvancement)
        assert adv.max_consecutive_losses == 3

    def test_negative_max_consecutive_losses_raises(self):
        with pytest.raises(ValueError, match="max_consecutive_losses"):
            parse_advancement(
                {
                    "strategy": "blocked_curricula",
                    "total_frame_budget": 100,
                    "max_consecutive_losses": -1,
                }
            )


# =========================================================================
# Fixed budget advancement (current behavior with new config shape)
# =========================================================================


class TestFixedBudgetAdvancement:
    """Verify fixed_budget strategy matches the existing Tomov 2023 protocol."""

    def test_advances_on_level_timeout(self, tmp_path):
        """Level budget expires -> advance regardless of win/loss."""
        # Budget = 27 = exactly one win path. After winning, budget exhausted -> advance.
        agent = _make_fixed_budget_agent(
            str(tmp_path), level_frame_budget=27, max_levels=2
        )
        agent.harness.mock_actions = list(WIN_L0) + list(WIN_L1)
        result = agent.run(start_level=0)

        assert result["cumulative_wins"] >= 1
        assert result["final_level"] >= 1

    def test_replays_level_when_budget_remains(self, tmp_path):
        """Win but budget not exhausted -> replay same level."""
        # Budget = 54 = two win paths. Agent wins twice on level 0, then budget exhausted.
        agent = _make_fixed_budget_agent(
            str(tmp_path), level_frame_budget=54, max_levels=1
        )
        agent.harness.mock_actions = list(WIN_L0) + list(WIN_L0)
        result = agent.run(start_level=0)

        assert result["cumulative_wins"] == 2

    def test_death_restarts_same_level(self, tmp_path):
        """Die -> restart same level, budget keeps ticking."""
        # Budget = 29 = 1 reset (1 frame) + 1 reset (1 frame) + 27 win steps
        agent = _make_fixed_budget_agent(
            str(tmp_path), level_frame_budget=29, max_levels=1
        )
        agent.harness.mock_actions = RESET + RESET + list(WIN_L0)
        result = agent.run(start_level=0)

        assert result["cumulative_wins"] == 1
        assert result["cumulative_losses"] == 2


# =========================================================================
# Blocked curricula advancement
# =========================================================================


class TestBlockedCurriculaAdvancement:
    """Verify blocked_curricula: advance on consecutive wins, global budget."""

    def test_two_consecutive_wins_advances(self, tmp_path):
        """Two consecutive wins on level 0 -> advance to level 1."""
        # 2 wins on L0 = 54 frames. Budget = 54: mastery at exactly 54, 0 left.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=54,
            consecutive_wins_required=2,
            max_levels=2,
        )
        agent.harness.mock_actions = list(WIN_L0) + list(WIN_L0)
        result = agent.run(start_level=0)

        assert result["final_level"] == 1
        assert result["cumulative_wins"] == 2

    def test_broken_streak_does_not_advance(self, tmp_path):
        """Win, die, win -> streak resets, no advancement."""
        # Each cycle: win(27) + die(1) = 28 frames. Budget = 84 allows 3 cycles.
        # Pattern: win-die-win-die-win-die -- never 2 consecutive wins.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=84,
            consecutive_wins_required=2,
            max_levels=2,
        )
        actions = (list(WIN_L0) + RESET) * 3
        agent.harness.mock_actions = actions
        result = agent.run(start_level=0)

        # Never achieved 2 consecutive wins, stayed on level 0
        assert result["final_level"] == 0
        assert result["cumulative_wins"] == 3
        assert result["cumulative_losses"] == 3

    def test_global_budget_ends_run(self, tmp_path):
        """Global budget expires before mastery -> run ends."""
        # Budget = 30 frames. One win takes 27 frames, only 3 remaining.
        # Can't complete second win -> budget expires.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=30,
            consecutive_wins_required=2,
            max_levels=2,
        )
        # One win (27 frames), then attempt another but budget runs out at frame 30
        agent.harness.mock_actions = list(WIN_L0) + list(WIN_L0)
        result = agent.run(start_level=0)

        assert result["final_level"] == 0
        assert result["cumulative_wins"] == 1

    def test_mastery_then_budget_expires_on_next_level(self, tmp_path):
        """Master level 0, then global budget expires during level 1."""
        # 2 wins on L0 = 54 frames. Budget = 70: 16 frames left for L1.
        # Not enough to win L1 (26 steps). Run ends mid-level-1.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=70,
            consecutive_wins_required=2,
            max_levels=9,
        )
        agent.harness.mock_actions = list(WIN_L0) + list(WIN_L0) + list(WIN_L1)
        result = agent.run(start_level=0)

        # Advanced to level 1 but couldn't finish it
        assert result["final_level"] == 1
        assert result["cumulative_wins"] == 2

    def test_single_consecutive_win_required(self, tmp_path):
        """consecutive_wins_required=1 -> any win advances immediately."""
        # win L0 (27) -> mastery -> advance. win L1 (26) -> mastery -> advance.
        # Total = 53 frames.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=53,
            consecutive_wins_required=1,
            max_levels=2,
        )
        agent.harness.mock_actions = list(WIN_L0) + list(WIN_L1)
        result = agent.run(start_level=0)

        assert result["final_level"] == 2
        assert result["cumulative_wins"] == 2
        assert result["outcome"] == "completed"

    def test_death_resets_consecutive_counter(self, tmp_path):
        """Die after one win -> streak resets. Then two clean wins -> advance."""
        # win(27) + die(1) + win(27) + win(27) = 82 frames
        # Budget = 82: mastery reached at exactly 82, then 0 frames left -> run ends
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=82,
            consecutive_wins_required=2,
            max_levels=2,
        )
        actions = list(WIN_L0) + RESET + list(WIN_L0) + list(WIN_L0)
        agent.harness.mock_actions = actions
        result = agent.run(start_level=0)

        # First win + death breaks streak. Then two consecutive wins -> advance.
        assert result["final_level"] == 1
        assert result["cumulative_wins"] == 3
        assert result["cumulative_losses"] == 1

    def test_mastery_across_multiple_levels(self, tmp_path):
        """Master both level 0 and level 1 with consecutive wins."""
        # L0: 2 wins = 54 frames. L1: 2 wins = 52 frames. Total = 106 frames.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=106,
            consecutive_wins_required=2,
            max_levels=2,
        )
        actions = (
            list(WIN_L0)
            + list(WIN_L0)  # master L0
            + list(WIN_L1)
            + list(WIN_L1)  # master L1
        )
        agent.harness.mock_actions = actions
        result = agent.run(start_level=0)

        assert result["final_level"] == 2
        assert result["cumulative_wins"] == 4
        assert result["outcome"] == "completed"


# =========================================================================
# Consecutive-losses early stop (blocked_curricula only)
# =========================================================================


class TestMaxConsecutiveLosses:
    """Verify max_consecutive_losses aborts stuck runs without burning budget."""

    def test_three_consecutive_deaths_abort(self, tmp_path):
        """3 consecutive deaths with mcl=3 -> run aborts before budget expires."""
        # 3 deaths = 3 frames consumed. Budget 1000 would let the run continue,
        # but the early stop fires after the 3rd death.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=1000,
            consecutive_wins_required=2,
            max_levels=2,
            max_consecutive_losses=3,
        )
        # Many RESETs queued; only the first 3 should execute before abort.
        agent.harness.mock_actions = RESET * 10
        result = agent.run(start_level=0)

        assert result["cumulative_losses"] == 3
        assert result["cumulative_wins"] == 0
        assert result["final_level"] == 0
        # Run ended via early stop, not budget exhaustion.
        assert result["total_steps"] == 3

    def test_win_resets_loss_counter(self, tmp_path):
        """A win between deaths resets the consecutive-loss counter."""
        # Pattern: die, die, WIN, die, die, die -> 6th action triggers abort.
        # (The 3rd death after the win is the 3rd-in-a-row.)
        # Frames used: 1+1+27+1+1+1 = 32. Budget 1000 leaves plenty of room.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=1000,
            consecutive_wins_required=2,
            max_levels=2,
            max_consecutive_losses=3,
        )
        agent.harness.mock_actions = (
            RESET + RESET + list(WIN_L0) + RESET + RESET + RESET + RESET * 5
        )
        result = agent.run(start_level=0)

        assert result["cumulative_wins"] == 1
        assert result["cumulative_losses"] == 5  # 2 before win + 3 after
        assert result["final_level"] == 0

    def test_disabled_by_default(self, tmp_path):
        """mcl=0 (default) lets a run die many times without aborting."""
        # 10 deaths then plenty of frame budget left. Without the early stop,
        # the run continues until budget expires.
        agent = _make_blocked_agent(
            str(tmp_path),
            total_frame_budget=10,  # 10 RESETs = 10 frames = exactly budget
            consecutive_wins_required=2,
            max_levels=2,
            max_consecutive_losses=0,
        )
        agent.harness.mock_actions = RESET * 10
        result = agent.run(start_level=0)

        # All 10 deaths executed; stopped by frame budget, not early stop.
        assert result["cumulative_losses"] == 10
        assert result["cumulative_wins"] == 0


# ===================================================================
# Oracle system prompt: {sprite_type} placeholder resolution
# ===================================================================


class TestOraclePromptSubstitution:
    """Verify that GameplayAgent resolves {sprite_type} placeholders in oracle
    system prompts using canonical colors from the game's sprite registry.

    The substitution happens on the first _register_and_make_env call, so we
    create an agent, trigger env creation, and check the live system prompt.
    """

    def _make_oracle_agent(self, tmp_path, game="bait_vgfmri4"):
        cfg = Config(
            harness=HarnessConfig(
                rationale_mode="action-only",
                suggestion_level="oracle",
            ),
            llm=LLMConfig(backend="mock", model=""),
            seed=0,
            game=GameConfig(
                game=game,
                start_level=0,
                max_levels=1,
                idle_frames=0,
                advancement={
                    "strategy": "fixed_budget",
                    "level_frame_budget": 10,
                },
            ),
            logging=LoggingConfig(
                output_dir=str(tmp_path), wandb_project="", verbose=False
            ),
        )
        validate_config(cfg)
        return GameplayAgent(cfg)

    def test_placeholders_present_before_env(self, tmp_path):
        """Before env creation, oracle prompt still has {sprite_type} placeholders."""
        agent = self._make_oracle_agent(tmp_path)
        assert "{avatar}" in agent.harness._system_prompt

    def test_placeholders_resolved_after_env(self, tmp_path):
        """After env creation, all {sprite_type} placeholders are resolved."""
        agent = self._make_oracle_agent(tmp_path)
        agent._register_and_make_env(level=0)
        sp = agent.harness._system_prompt
        assert "{avatar}" not in sp
        assert "{key}" not in sp
        assert "{goal}" not in sp
        assert "{game_rules}" not in sp
        assert "{max_output_tokens}" not in sp

    def test_canonical_colors_injected(self, tmp_path):
        """Resolved prompt contains the canonical colors from the game definition."""
        agent = self._make_oracle_agent(tmp_path)
        agent._register_and_make_env(level=0)
        sp = agent.harness._system_prompt
        # bait_vgfmri4 canonical: avatar=YELLOW, key=ORANGE, goal=PURPLE
        assert "YELLOW" in sp
        assert "ORANGE" in sp
        assert "PURPLE" in sp

    def test_conversation_system_msg_updated(self, tmp_path):
        """The live conversation[0] must match the resolved system prompt."""
        agent = self._make_oracle_agent(tmp_path)
        agent._register_and_make_env(level=0)
        assert agent.harness._conversation[0]["content"] == agent.harness._system_prompt

    @pytest.mark.parametrize(
        "game",
        [
            "avoidGeorge_vgfmri4",
            "bait_vgfmri3",
            "bait_vgfmri4",
            "chase_vgfmri3",
            "chase_vgfmri4",
            "helper_vgfmri3",
            "helper_vgfmri4",
            "lemmings_vgfmri3",
            "lemmings_vgfmri4",
            "plaqueAttack_vgfmri3",
            "sokoban_vgfmri3",
            "zelda_vgfmri3",
            "zelda_vgfmri4",
        ],
    )
    def test_all_games_resolve(self, tmp_path, game):
        """Every game resolves all sprite-type placeholders."""
        import re

        agent = self._make_oracle_agent(tmp_path, game=game)
        agent._register_and_make_env(level=0)
        sp = agent.harness._system_prompt
        leftover = re.findall(r"\{(\w+)\}", sp)
        real_placeholders = [p for p in leftover if p not in ("action", "rationale")]
        assert real_placeholders == [], f"unresolved placeholders: {real_placeholders}"
