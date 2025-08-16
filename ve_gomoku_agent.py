import json
import re
from typing import List, Tuple
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VEGomukuAgent(Agent):
    """LLM-first Gomoku agent: rely on model judgment; code only validates and falls back.

    Goal: maximize LLM usage while keeping responses valid, fast, and legal.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Created VEGomukuAgent: {agent_id}")

    def _setup(self):
        self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        return (
            "You are an elite Gomoku (8x8, five-in-a-row) analyst and player.\n"
            "Think carefully but output ONLY JSON.\n\n"
            "Hard priorities (in order):\n"
            "1) WIN-IN-1: If any move makes five in a row (row/col/diagonals \\ and //), pick it.\n"
            "2) BLOCK-IN-1: If opponent can win next move, block that exact square.\n"
            "3) CREATE POWER THREATS: prefer open-fours or two simultaneous winning threats; extend strongest lines near center.\n"
            "4) AVOID BLUNDERS: don't choose a move that allows an immediate opponent win next move.\n"
            "5) TIE-BREAKERS: prefer central/backbone squares and moves that create forks (multiple threats).\n\n"
            "Process: scan with 5-length windows across rows, columns, and both diagonals; list candidate squares and choose best.\n"
            "Coordinates are 0-indexed. You MUST choose from legal_moves.\n\n"
            "Return STRICT JSON only: {\"row\": <int>, \"col\": <int>}\n"
        )

    # --- Minimal safeguards: helpers ---
    def _parse_board_from_string(self, board_str: str) -> List[List[str]]:
        rows: List[List[str]] = []
        for line in board_str.strip().split('\n'):
            tokens = [ch for ch in line if ch in ['X', 'O', '.']]
            if len(tokens) == 8:
                rows.append(tokens)
        while len(rows) < 8:
            rows.append(['.'] * 8)
        return rows[:8]

    def _five_in_row_if_place(self, board: List[List[str]], r: int, c: int, player: str) -> bool:
        if board[r][c] != '.':
            return False
        board[r][c] = player
        win = False
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == player:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == player:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= 5:
                win = True
                break
        board[r][c] = '.'
        return win

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        # Prepare inputs
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

        # Minimal safeguards: immediate win-in-1, then block-in-1
        try:
            board = self._parse_board_from_string(board_str)
            # 1) Win now if possible (choose most-central among winning moves)
            winning = [(r,c) for (r,c) in legal_moves if self._five_in_row_if_place(board, r, c, me)]
            if winning:
                winning.sort(key=lambda rc: (abs(rc[0]-3.5)+abs(rc[1]-3.5), rc))
                return winning[0]
            # 2) Block opponent's immediate win squares (most-central if multiple)
            must_block: List[Tuple[int,int]] = []
            for rr in range(8):
                for cc in range(8):
                    if board[rr][cc] == '.' and self._five_in_row_if_place(board, rr, cc, opp):
                        must_block.append((rr, cc))
            blocks = [m for m in legal_moves if m in must_block]
            if blocks:
                blocks.sort(key=lambda rc: (abs(rc[0]-3.5)+abs(rc[1]-3.5), rc))
                return blocks[0]
        except Exception:
            pass

        # Primary LLM selection
        primary_prompt = (
            f"Player: {me} (opponent: {opp})\n"
            f"Board (standard):\n{board_str}\n\n"
            f"legal_moves (choose exactly one of these): {legal_moves}\n\n"
            "Decide using the priorities above. Return JSON only from legal_moves."
        )
        move = await self._llm_pick(primary_prompt, legal_moves, temperature=0.1)
        if move is not None:
            return move

        # Repair attempt: emphasize legality only
        repair_prompt = (
            f"Pick ONE move strictly from legal_moves: {legal_moves}.\n"
            "Return ONLY JSON: {\"row\": <int>, \"col\": <int>}"
        )
        move = await self._llm_pick(repair_prompt, legal_moves, temperature=0.0)
        if move is not None:
            return move

        # Final minimal fallback: most central
        return min(legal_moves, key=lambda rc: (abs(rc[0]-3.5) + abs(rc[1]-3.5), rc))

    async def _llm_pick(self, user_prompt: str, legal_moves: List[Tuple[int, int]], temperature: float = 0.1) -> Tuple[int, int] | None:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            resp = await self.llm.complete(messages=messages, temperature=temperature, max_tokens=160)
            m = re.search(r"\{[^}]+\}", resp, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            move = (int(data.get("row", -1)), int(data.get("col", -1)))
            return move if move in legal_moves else None
        except Exception:
            return None
