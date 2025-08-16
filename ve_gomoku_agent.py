import json
import re
from typing import List, Tuple
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VEGomukuAgent(Agent):
    """LLM-first Gomoku agent: minimal rules, rely on the LLM for tactics and planning.

    Philosophy:
    - Use the LLM to perform win/block/threat analysis and choose a move.
    - Keep code-side logic to validation and a simple fallback only.
    - Strict JSON-only I/O to avoid parsing issues.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Created VEGomukuAgent: {agent_id}")

    def _setup(self):
        self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        return (
            "You are an elite Gomoku (8x8, five-in-a-row) strategist.\n"
            "Decide the best move by analyzing the full board (rows, columns, diagonals \\ and //).\n"
            "Hard priorities: (1) win-in-1, (2) block-in-1, (3) create/open fours and double threats,\n"
            "(4) prevent opponent immediate win next move, (5) prefer central/backbone extensions and forks.\n"
            "You MUST pick one coordinate from legal_moves. Coordinates are 0-indexed.\n\n"
            "Output format (STRICT): JSON only, nothing else: {\"row\": <int>, \"col\": <int>}\n"
        )

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        # Collect inputs
        try:
            board_str = game_state.format_board(formatter="standard")
        except Exception:
            board_str = ""
        try:
            legal_moves: List[Tuple[int, int]] = list(game_state.get_legal_moves())
        except Exception:
            legal_moves = []
        if not legal_moves:
            return (4, 4)

        me = getattr(getattr(game_state, "current_player", object()), "value", "?")
        opp = 'O' if me == 'X' else 'X'

        # Primary LLM call
        user_prompt = (
            f"Player: {me} (opponent: {opp})\n"
            f"Board:\n{board_str}\n\n"
            f"legal_moves (choose exactly one of these): {legal_moves}\n\n"
            "Instructions (concise):\n"
            "- First check: can you win now with 5-in-a-row? If yes, play it.\n"
            "- Then check: can opponent win next move? Block that exact square.\n"
            "- Prefer moves creating open-fours or two simultaneous winning threats.\n"
            "- Avoid any move that lets opponent create 5 next.\n"
            "Return JSON only from legal_moves."
        )
        move = await self._ask_llm_for_move(user_prompt, legal_moves)
        if move is not None:
            return move

        # One-shot repair: ask LLM again to pick from legal_moves if previous answer illegal/invalid
        repair_prompt = (
            f"Your previous selection was invalid or not in legal_moves.\n"
            f"Choose ONE coordinate from legal_moves only: {legal_moves}.\n"
            "Return JSON only: {\"row\": <int>, \"col\": <int>}"
        )
        move = await self._ask_llm_for_move(repair_prompt, legal_moves)
        if move is not None:
            return move

        # Final minimal fallback: most central legal move
        return min(legal_moves, key=lambda rc: (abs(rc[0]-3.5) + abs(rc[1]-3.5), rc))

    async def _ask_llm_for_move(self, prompt: str, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int] | None:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            resp = await self.llm.complete(messages=messages, temperature=0.1, max_tokens=120)
            m = re.search(r"\{[^}]+\}", resp, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            move = (int(data.get("row", -1)), int(data.get("col", -1)))
            return move if move in legal_moves else None
        except Exception:
            return None
