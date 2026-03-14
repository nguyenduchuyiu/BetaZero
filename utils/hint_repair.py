from collections import defaultdict
from tqdm import tqdm
import re
import sys

class ProofRepairer:
    def __init__(self, code: str, verifier, verbose=True):
        self.code = code
        self.verifier = verifier
        self.verbose = verbose
        self.try_repairer = 'try norm_cast ; try norm_num ; try simp_all ; try ring_nf at * ; try native_decide ; try linarith ; try nlinarith ; try aesop'
    def repair_proof(self) -> str:
        code_with_hints = self.code.replace('sorry', 'hint')
        if self.verbose:
            print('Begin HintRepair...')
        err_info = self.verifier(code_with_hints)

        if 'infos' in err_info:
            hint_correct = []
            for i, info in enumerate(err_info['infos'], start=1):
                try:
                    suggestions = self.get_hint_tactics(info['data'])
                except Exception:
                    suggestions = []
                if len(suggestions) == 1:
                    hint_correct.append([i, suggestions[0]])

            replacement_accumulation = 0
            iterator = tqdm(hint_correct, desc='Correcting with hint') if self.verbose else hint_correct
            for idx, tactic in iterator:
                idx -= replacement_accumulation
                self.code = self.replace_nth('sorry', tactic, self.code, idx)
                replacement_accumulation += 1

        def count_unsolved(err_dict):
            return sum(1 for e in err_dict.get('errors', []) if 'unsolved goals' in e['data'])
            
        def count_fatal(err_dict):
            return sum(1 for e in err_dict.get('errors', []) if 'unsolved goals' not in e['data'])

        base_err = self.verifier(self.code)
        if base_err.get('pass', False):
            return self.code

        base_unsolved = count_unsolved(base_err)
        base_fatal = count_fatal(base_err)

        total_sorries = self.code.count('sorry')
        if total_sorries == 0:
            return self.code

        if self.verbose:
            print('Begin Hint Repair...')
            
        pbar = tqdm(total=total_sorries, desc='Correcting with other solvers') if self.verbose else None
        attempt_idx = 1
        
        while True:
            parts = self.code.split('sorry')
            if attempt_idx >= len(parts):
                break

            part1 = 'sorry'.join(parts[:attempt_idx])
            part2 = 'sorry'.join(parts[attempt_idx:])
            test_code = part1 + self.try_repairer + part2

            test_err = self.verifier(test_code)
            test_unsolved = count_unsolved(test_err)
            test_fatal = count_fatal(test_err)

            if test_err.get('pass', False):
                self.code = test_code
                if pbar: pbar.update(len(parts) - attempt_idx)
                break

            if test_fatal <= base_fatal and test_unsolved < base_unsolved:
                self.code = test_code
                base_unsolved = test_unsolved
                base_fatal = test_fatal
            else:
                attempt_idx += 1
            
            if pbar: pbar.update(1)

        if pbar: pbar.close()
        return self.code

    def replace_nth(self, sub, repl, txt, nth):
        arr = txt.split(sub)
        part1 = sub.join(arr[:nth])
        part2 = sub.join(arr[nth:])
        return part1 + repl + part2

    def get_hint_tactics(self, info):
        lines = info.splitlines()
        suggestions = []
        current_suggestion = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("•"):
                if current_suggestion is not None:
                    suggestions.append(current_suggestion)
                current_suggestion = stripped[1:].strip()
            else:
                if current_suggestion is not None and stripped:
                    current_suggestion += " " + stripped
        if current_suggestion is not None and current_suggestion.strip() not in ('aesop', 'intro'):
            suggestions.append(current_suggestion)
        return suggestions

    def verify_lean_code(self, code):
        return self.verifier(code)

class LeanServerProofRepairer(ProofRepairer):
    def verify_lean_code(self, code):
        request_id = self.verifier.verifier_submit_request(code)
        result_list = self.verifier.verifier_get_all_request_outputs([request_id])
        return result_list[0]