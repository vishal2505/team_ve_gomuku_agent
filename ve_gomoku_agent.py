import json
import re
import random
import os
from typing import Tuple, List
from gomoku import Agent, GameState
from gomoku.llm import OpenAIGomokuClient

class VEGomukuAgent(Agent):
    """LLM-powered Gomoku agent with stronger diagonal threat capture and blunder-avoidance."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"Created VEGomukuAgent: {agent_id}")

    def _setup(self):
        # Configure OpenAI API settings
        #os.environ["OPENAI_BASE_URL"] = "https://api.mtkachenko.info/v1"
        #os.environ["OPENAI_API_KEY"] = ""
        
        self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="google/gemma-2-9b-it")

    def _create_system_prompt(self) -> str:
        return """
You are a TOURNAMENT-WINNING Gomoku Grand Master. Your goal is TOTAL VICTORY.

=== ABSOLUTE PRIORITIES (Check in this order) ===
1. IMMEDIATE WIN: If you can create 5-in-a-row, DO IT NOW
2. BLOCK OPPONENT WIN: If opponent has 4-in-a-row + empty space, BLOCK IT
3. BLOCK CRITICAL THREATS: Stop opponent's dangerous patterns immediately
4. CREATE WINNING ATTACKS: Build unstoppable multiple threats
5. STRATEGIC POSITIONING: Control center and key intersections

=== CRITICAL THREAT PATTERNS TO BLOCK ===
- Open 4: "XXXX_" or "_XXXX" → BLOCK IMMEDIATELY
- Closed 4: "_XXXX_" → BLOCK one end
- Open 3: "_XXX_" → EXTREMELY DANGEROUS, block one end
- Broken 3: "X_XX_", "_X_XX", "XX_X_" → Block the gap
- Double 3: Two 3-patterns crossing → HIGHEST PRIORITY TO BLOCK

=== WINNING ATTACK PATTERNS TO CREATE ===
- Double 4: Two 4-patterns = INSTANT WIN
- 4+3 Fork: 4-pattern + 3-pattern = GUARANTEED WIN  
- Double 3: Two open 3-patterns = VERY STRONG
- 3+3 Fork: Multiple 3-patterns crossing

=== ADVANCED TACTICS ===
- FORKS: Create positions where you threaten multiple ways to win
- FORCING MOVES: Make opponent respond to your threats
- TEMPO: Each move should either threaten or improve your position
- SACRIFICE: Sometimes give up material to create unstoppable attack

=== POSITIONAL GUIDELINES ===
- Early game (moves 1-10): Stay within 3 spaces of center (3,3) or (4,4)
- NEVER play corners unless it wins/blocks immediately
- AVOID edges early unless forced
- Connect your stones when possible
- Control diagonals - they're hardest for opponent to see

=== BLUNDER PREVENTION ===
Before choosing any move, VERIFY:
- Does this move let opponent win immediately next turn? If YES, DON'T PLAY IT
- Does this move block opponent's most dangerous threat?
- Does this move create my strongest possible attack?

=== TOURNAMENT MINDSET ===
- ASSUME opponent is expert-level
- EVERY move must have purpose
- CALCULATE 2-3 moves ahead
- PREFER aggressive attacking over passive defense
- When equal options exist, choose the one that creates MORE threats

Return ONLY this JSON format:
{"reasoning": "Brief tactical explanation of why this move wins/blocks/attacks", "row": <int>, "col": <int>}
""".strip()

    def _parse_board_from_string(self, board_str: str) -> List[List[str]]:
        rows = []
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
        won = False
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
                won = True
                break
        board[r][c] = '.'
        return won

    def _move_gives_opp_immediate_win(self, board: List[List[str]], move: Tuple[int, int], me: str, opp: str) -> bool:
        r, c = move
        if board[r][c] != '.':
            return True
        board[r][c] = me
        # After my move, if opponent has any winning reply, this is a blunder.
        for rr in range(8):
            for cc in range(8):
                if board[rr][cc] == '.' and self._five_in_row_if_place(board, rr, cc, opp):
                    board[r][c] = '.'
                    return True
        board[r][c] = '.'
        return False

    def _check_line_for_threat(self, board: List[List[str]], start_r: int, start_c: int, dr: int, dc: int, player: str, target_count: int) -> List[Tuple[int, int]]:
        threat_positions = []
        line = []
        r, c = start_r, start_c
        while 0 <= r - dr < 8 and 0 <= c - dc < 8:
            r -= dr
            c -= dc
        while 0 <= r < 8 and 0 <= c < 8:
            line.append((r, c))
            r += dr
            c += dc
        opp = 'O' if player == 'X' else 'X'
        # 5-window scan
        for i in range(0, max(0, len(line) - 4)):
            window = line[i:i+5]
            pieces = [board[rr][cc] for rr, cc in window]
            pc = pieces.count(player)
            ec = pieces.count('.')
            oc = pieces.count(opp)
            if target_count == 4 and pc == 4 and ec == 1 and oc == 0:
                threat_positions.append(window[pieces.index('.')])
                continue
            if target_count == 3 and pc == 3 and ec == 2 and oc == 0:
                # Contiguous .XXX.
                if pieces[0] == '.' and pieces[1] == player and pieces[2] == player and pieces[3] == player and pieces[4] == '.':
                    threat_positions.extend([window[0], window[4]])
                else:
                    # Broken threes X.XX or XX.X
                    for idx, v in enumerate(pieces):
                        if v == '.':
                            threat_positions.append(window[idx])
                continue
            if target_count == 3 and pc == 3 and ec == 1 and oc == 1:
                threat_positions.append(window[pieces.index('.')])
        # 4-window scan for edges (.XXX / XXX.)
        if target_count == 3 and len(line) >= 4:
            for i in range(0, len(line) - 3):
                window = line[i:i+4]
                pieces = [board[rr][cc] for rr, cc in window]
                pc = pieces.count(player)
                ec = pieces.count('.')
                oc = pieces.count(opp)
                if pc == 3 and ec == 1 and oc == 0:
                    if pieces[0] == '.':
                        threat_positions.append(window[0])
                    if pieces[3] == '.':
                        threat_positions.append(window[3])
        return threat_positions

    def _find_all_threats(self, board: List[List[str]], player: str, target_count: int) -> List[Tuple[int, int]]:
        threats = set()
        for r in range(8):
            for c in range(8):
                for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                    threats.update(self._check_line_for_threat(board, r, c, dr, dc, player, target_count))
        return list(threats)

    def _score_move(self, board: List[List[str]], r: int, c: int, me: str) -> int:
        if board[r][c] != '.':
            return -1
        board[r][c] = me
        best = 0
        forks = 0
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            cnt = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == me:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == me:
                cnt += 1
                rr -= dr
                cc -= dc
            best = max(best, cnt)
            if cnt >= 3:
                forks += 1
        board[r][c] = '.'
        return best * 10 + forks

    def _center_dist(self, pos: Tuple[int, int]) -> float:
        return abs(pos[0] - 3.5) + abs(pos[1] - 3.5)
    
    def _analyze_board_for_llm(self, board: List[List[str]], me: str, opp: str) -> str:
        """Provide detailed tactical analysis for LLM context"""
        analysis = []
        
        # Check for immediate wins
        my_wins = [(r,c) for r in range(8) for c in range(8) 
                  if board[r][c] == '.' and self._five_in_row_if_place(board, r, c, me)]
        if my_wins:
            analysis.append(f"IMMEDIATE WIN AVAILABLE: {my_wins}")
        
        # Check opponent wins to block
        opp_wins = [(r,c) for r in range(8) for c in range(8) 
                   if board[r][c] == '.' and self._five_in_row_if_place(board, r, c, opp)]
        if opp_wins:
            analysis.append(f"MUST BLOCK OPPONENT WIN: {opp_wins}")
        
        # Find my threats
        my_4threats = self._find_all_threats(board, me, 4)
        my_3threats = self._find_all_threats(board, me, 3)
        if my_4threats:
            analysis.append(f"MY 4-THREATS: {my_4threats}")
        if my_3threats:
            analysis.append(f"MY 3-THREATS: {my_3threats}")
        
        # Find opponent threats  
        opp_4threats = self._find_all_threats(board, opp, 4)
        opp_3threats = self._find_all_threats(board, opp, 3)
        if opp_4threats:
            analysis.append(f"BLOCK OPP 4-THREATS: {opp_4threats}")
        if opp_3threats:
            analysis.append(f"BLOCK OPP 3-THREATS: {opp_3threats}")
        
        # Identify dangerous moves
        legal = [(r,c) for r in range(8) for c in range(8) if board[r][c] == '.']
        blunders = [m for m in legal if self._move_gives_opp_immediate_win(board, m, me, opp)]
        if blunders:
            analysis.append(f"AVOID BLUNDERS: {blunders}")
        
        return "\n".join(analysis) if analysis else "No critical threats detected."

    def _pick_best(self, candidates: List[Tuple[int, int]], legal: List[Tuple[int, int]], board: List[List[str]], me: str, opp: str, avoid_blunders: bool = True) -> Tuple[int, int] | None:
        moves = [m for m in candidates if m in legal]
        if not moves:
            return None
        if avoid_blunders:
            safe = [m for m in moves if not self._move_gives_opp_immediate_win(board, m, me, opp)]
            if safe:
                moves = safe
        return min(moves, key=lambda p: (-self._score_move(board, p[0], p[1], me), self._center_dist(p)))

    def _get_strategic_move(self, board: List[List[str]], me: str, opp: str, legal: List[Tuple[int, int]]) -> Tuple[int, int] | None:
        # 1. Win now
        win_moves = [(r,c) for (r,c) in legal if self._five_in_row_if_place(board, r, c, me)]
        best = self._pick_best(win_moves, legal, board, me, opp, avoid_blunders=False)
        if best:
            return best
        # 2. Block opponent win
        block_win = self._find_all_threats(board, opp, 4)
        best = self._pick_best(block_win, legal, board, me, opp)
        if best:
            return best
        # 3. Block opponent open/broken threes
        block_three = self._find_all_threats(board, opp, 3)
        best = self._pick_best(block_three, legal, board, me, opp)
        if best:
            return best
        # 4. Create our threes
        my_three = self._find_all_threats(board, me, 3)
        best = self._pick_best(my_three, legal, board, me, opp)
        if best:
            return best
        # 5. Early-game center bias and safety
        move_count = sum(row.count('X') + row.count('O') for row in board)
        if move_count < 10:
            ring = [m for m in legal if self._center_dist(m) <= 3]
            ring_safe = [m for m in ring if not self._move_gives_opp_immediate_win(board, m, me, opp)]
            pool = ring_safe or ring
            if pool:
                return min(pool, key=lambda p: (-self._score_move(board, p[0], p[1], me), self._center_dist(p)))
        # 6. Otherwise pick safest best-scoring
        safe_all = [m for m in legal if not self._move_gives_opp_immediate_win(board, m, me, opp)]
        pool = safe_all or legal
        return min(pool, key=lambda p: (-self._score_move(board, p[0], p[1], me), self._center_dist(p)))

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        try:
            me = game_state.current_player.value
            opp = 'O' if me == 'X' else 'X'
            legal = game_state.get_legal_moves()
            board_str = game_state.format_board(formatter="standard")
            board = self._parse_board_from_string(board_str)
            
            move_count = sum(row.count('X') + row.count('O') for row in board)
            
            # Get detailed tactical analysis
            tactical_analysis = self._analyze_board_for_llm(board, me, opp)
            
            # Build comprehensive user prompt with all strategic context
            user_prompt = f"""
=== CURRENT GAME STATE ===
You: {me} | Opponent: {opp} | Move #{move_count + 1}
Game Phase: {"Opening" if move_count < 8 else "Middle" if move_count < 20 else "Endgame"}

=== BOARD POSITION ===
{board_str}

=== TACTICAL ANALYSIS ===
{tactical_analysis}

=== LEGAL MOVES ===
Available positions: {legal}

=== YOUR MISSION ===
1. If IMMEDIATE WIN exists → Take it NOW
2. If opponent can win next → BLOCK IT  
3. If opponent has dangerous threats → BLOCK THEM
4. Otherwise → Create your STRONGEST ATTACK

=== STRATEGIC GUIDELINES FOR THIS POSITION ===
- Move #{move_count + 1} {"focus on CENTER control" if move_count < 8 else "BUILD THREATS and FORKS" if move_count < 20 else "CALCULATE FOR WIN"}
- Look for DOUBLE THREATS (where you threaten 2+ ways to win)
- PREFER moves that force opponent to respond defensively
- CHECK: Does your move give opponent immediate win? If yes, don't play it!

=== EVALUATION CRITERIA ===
Best move should:
✓ Block all opponent wins/threats
✓ Create maximum threats for you  
✓ NOT allow opponent counter-attack
✓ Control key board positions
✓ Force opponent into defensive play

ANALYZE the position step-by-step, then return your best move.
JSON format only:
{{"reasoning": "Step-by-step analysis: 1) Check wins/blocks 2) Evaluate threats 3) Choose best attack", "row": <int>, "col": <int>}}
"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            response = await self.llm.complete(messages=messages)
            
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                data = json.loads(response[start:end])
                r, c = int(data["row"]), int(data["col"])
                if (r, c) in legal:
                    # Final safety check - avoid obvious blunders
                    if not self._move_gives_opp_immediate_win(board, (r, c), me, opp):
                        return (r, c)
                    else:
                        print(f"LLM suggested blunder {(r,c)}, finding safe alternative...")
            
        except Exception as e:
            print(f"LLM Agent error: {e}")
        
        # Emergency fallback with safety check
        safe = [m for m in legal if not self._move_gives_opp_immediate_win(board, m, me, opp)] if 'board' in locals() else []
        pool = safe or legal
        return min(pool, key=lambda p: self._center_dist(p))