from tokenizers import Tokenizer

def main():
    token = Tokenizer.from_file("tokenizer.json")

    token.enable_padding()
    token.enable_truncation(max_length=128)

    # Single sentence — Sherlock Holmes
    text = "Mr. Holmes, they were the footprints of a gigantic hound."
    # Pair — Shakespearean style pair
    text_a = "To be, or not to be, that is the question."
    text_b = "Whether ’tis nobler in the mind to suffer the slings and arrows of outrageous fortune."
    # Batch — Tolstoy passages
    batch = [
        "Well, Prince, so Genoa and Lucca are now just family estates of the Buonapartes.",
        "He looked round the salon with satisfaction, his eyes resting on the faces of the ladies."
    ]

    # === SINGLE ENCODE ===
    enc = token.encode(text)
    print("\n--- SINGLE ENCODE (Baskervilles) ---")
    print("Input:", text)
    print("Tokens:", enc.tokens)
    print("IDs:", enc.ids)
    print("Attention mask:", enc.attention_mask)

    # === PAIR ENCODE ===
    enc_pair = token.encode(text_a, text_b)
    print("\n--- PAIR ENCODE (Shakespeare) ---")
    print("Tokens:", enc_pair.tokens)
    print("IDs:", enc_pair.ids)

    # === BATCH ENCODE ===
    encs = token.encode_batch(batch)
    print("\n--- BATCH ENCODE (War and Peace) ---")
    for i, e in enumerate(encs):
        print(f"Input {i+1}: {batch[i]}")
        print("  Tokens:", e.tokens)
        print("  IDs:", e.ids)
        print("  Attention:", e.attention_mask)

if __name__ == "__main__":
    main()