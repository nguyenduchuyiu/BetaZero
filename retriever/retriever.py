import pickle
import os
import torch
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTextEncoding


class RAGRetriever:
    """Utility class for retrieving premises from proof states."""
    
    def __init__(
        self,
        indexed_corpus_path: Optional[str] = None,
        model_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small",
        device: Optional[torch.device] = None
    ):
        """Initialize the RAG retriever.
        
        Args:
            indexed_corpus_path: Path to the indexed corpus pickle file.
                If None, defaults to rag_module/indexed_corpus.pkl relative to this file.
            model_name: HuggingFace model name for the retriever
            device: PyTorch device (defaults to cuda if available, else cpu)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        
        # Set default path if not provided
        if indexed_corpus_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            indexed_corpus_path = os.path.join(module_dir, "indexed_corpus.pkl")
        
        # Load indexed corpus (plain data: dict with "premises" and "embeddings")
        print(f"Loading indexed corpus from {indexed_corpus_path}...")
        with open(indexed_corpus_path, "rb") as f:
            data = pickle.load(f)
        
        self.premises = data["premises"]  # list of dicts
        self.corpus_embeddings = data["embeddings"].to(device)
        
        print(f"Loaded {len(self.premises)} premises")
        
        # Load retriever model
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTextEncoding.from_pretrained(model_name).to(device)
        self.model.eval()
        print("Model loaded successfully")
    
    @torch.no_grad()
    def _encode(self, s: Union[str, List[str]]) -> torch.Tensor:
        """Encode text(s) into feature vectors."""
        if isinstance(s, str):
            s = [s]
            should_squeeze = True
        else:
            should_squeeze = False
        
        tokenized_s = self.tokenizer(s, return_tensors="pt", padding=True).to(self.device)
        hidden_state = self.model(tokenized_s.input_ids).last_hidden_state
        lens = tokenized_s.attention_mask.sum(dim=1)
        features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
        
        if should_squeeze:
            features = features.squeeze()
        return features
    
    def retrieve(
        self,
        proof_state: str,
        k: int = 10
    ) -> List[Tuple[dict, float]]:
        """Retrieve top-k premises for a given proof state.
        
        Args:
            proof_state: The proof state string (e.g., "x y : ℝ\n⊢ x + y = y + x")
            k: Number of top premises to retrieve
            
        Returns:
            List of tuples (premise_dict, similarity_score) sorted by score descending.
            Each premise_dict has keys: path, full_name, start, end, code.
        """
        state_emb = self._encode(proof_state)
        scores = (state_emb @ self.corpus_embeddings.T)
        topk_scores, topk_indices = scores.topk(k)
        
        results = []
        for i, idx in enumerate(topk_indices.tolist()):
            results.append((self.premises[idx], topk_scores[i].item()))
        
        return results


# Global retriever instance (lazy initialization)
_retriever_instance: Optional[RAGRetriever] = None


def get_retriever(
    indexed_corpus_path: Optional[str] = None,
    model_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small",
    device: Optional[torch.device] = None
) -> RAGRetriever:
    """Get or create the global retriever instance.
    
    Args:
        indexed_corpus_path: Path to the indexed corpus pickle file
        model_name: HuggingFace model name for the retriever
        device: PyTorch device
        
    Returns:
        RAGRetriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever(indexed_corpus_path, model_name, device)
    return _retriever_instance


def retrieve(
    proof_state: str,
    k: int = 10,
    indexed_corpus_path: Optional[str] = None,
    model_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small",
    device: Optional[torch.device] = None
) -> List[Tuple[dict, float]]:
    """Convenience function to retrieve premises for a proof state.
    
    This function automatically initializes the retriever on first call.
    
    Args:
        proof_state: The proof state string
        k: Number of top premises to retrieve
        indexed_corpus_path: Path to the indexed corpus pickle file
        model_name: HuggingFace model name for the retriever
        device: PyTorch device
        
    Returns:
        List of tuples (premise, similarity_score) sorted by score descending
    """
    retriever = get_retriever(indexed_corpus_path, model_name, device)
    return retriever.retrieve(proof_state, k)

if __name__ == "__main__":
    proof_state = """  x y : ℝ
    ho : (x + y) / (2 : ℝ) = (7 : ℝ)
    h₁ : √(x * y) = √(19 : ℝ)
    sum : x + y = (14 : ℝ)
    prod_pos : (0 : ℝ) < x * y
    prod : x * y = (19 : ℝ)
    ⊢ x ^ (2 : ℕ) + y ^ (2 : ℕ) = (158 : ℝ)"""

    # Retrieve top-10 premises
    results = retrieve(proof_state, k=10)

    # Print results
    print(f"\nRetrieved {len(results)} premises:")
    for i, (premise, score) in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {score:.4f}")
        print(f"Name: {premise['full_name']}")
        print(premise["code"][:200] + ("..." if len(premise["code"]) > 200 else ""))