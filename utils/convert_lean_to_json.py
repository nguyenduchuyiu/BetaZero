import json
import pickle
import glob
from tqdm import tqdm
import re
import os

import numpy as np
    
    

def clean_set_option(text):
    if 'set_option' in text:
        return ''
    else:
        return text

def remove_repeating_lemmas(lst):
    t = set()
    temp = []
    for l in lst:
        if l['name'] not in t:
            t.add(l['name'])
            temp.append(l)
        else:
            print(l['name'])
    return temp
        

def remove_comment(text):
    idx = text.find('--')
    if idx >= 0:
        return text[:idx]
    else:
        return text

def extract_proofs(txt, clean_comment_text=False):
    # 1. Vệ sinh văn bản (giữ nguyên code cũ đã hoạt động tốt)
    txt = re.sub(r'\xc2\xa0', ' ', txt)
    txt = re.sub(r'\xa0', ' ', txt)
    
    # 2. Split (giữ nguyên logic split mềm dẻo)
    blocks = re.split(r'(?m)(?=^[\s\xc2\xa0]*(?:lemma|theorem)\b)', txt)
    header = blocks[0] if len(blocks) > 1 else ""
    
    lst = []
    
    # Xử lý trường hợp file chỉ có 1 block (header dính liền theorem)
    iterable_blocks = blocks[1:] if len(blocks) > 1 else blocks

    for block in iterable_blocks:
        # Tìm xem đây là theorem hay lemma
        # Lưu lại match object để lấy vị trí bắt đầu (start index)
        m_start = re.search(r'\b(?:lemma|theorem)\s+([^\s(:]+)', block)
        if not m_start:
            continue
            
        name = m_start.group(1)
        
        # --- FIX QUAN TRỌNG TẠI ĐÂY ---
        # Lấy vị trí bắt đầu của chữ 'theorem'/'lemma'
        # Để loại bỏ các dòng import/set_option thừa ở phía trước nếu có
        start_index = m_start.start() 
        # ------------------------------
        block_header = header if header else block[:start_index]

        if clean_comment_text: 
            block = '\n'.join([remove_comment(l) for l in block.split('\n') if remove_comment(l).strip() != ''])

        m_by = re.search(r':=\s*by\b', block)
        if not m_by:
            continue

        lst.append({
            'name': name,
            'split': 'train',
            'header': block_header,
            'informal_prefix': '',
            # Cắt từ start_index thay vì cắt từ đầu
            'formal_statement': block[start_index:m_by.end()].strip(),
            'formal_proof': block[m_by.end():].strip(),
        })

    return lst
        

def get_deepseek_format_proofs(lemmas):
    save_arr = []
    for lemma in lemmas:
        t = extract_proofs(lemma, clean_comment_text=True)
        save_arr.extend(t)
    return save_arr


        