# From internet text to an AI assistant

Assignment 2, part 4 – AI in Accounting and Business (SDU).
All files from the seven practice steps. Everything in this repository comes from my own runs.

## Files by step

| Step | What I did | Files |
|---|---|---|
| 1. Training data | Downloaded 10 FineWeb documents and joined their text into one continuous block | `fineweb_sample.json`, `fineweb_corpus.txt` |
| 2. Text representation | Converted the first sentence to bits and bytes (CyberChef) and the whole corpus to GPT-4o token IDs (Tiktokenizer): 4,731 tokens | `fineweb_bits.txt`, `fineweb_bytes.txt`, `fineweb_tokens.txt` |
| 3. Train a small model | Trained a tiny transformer on my Mac for 60 seconds (2,526 steps, loss from about 7 to 0.04), then tested it | `requirements.txt`, `train.py`, `training_log.csv`, `loss_plot.png`, `samples.txt`, `model.pt`, `sky_test.py`, `sky_completions.txt`, `corpus_test.py`, `corpus_completions.txt` |
| 4. Base model and assistant | Compared SmolLM2-135M (base) with SmolLM2-135M-Instruct, prompted the base model with an example conversation, and tested the knowledge cutoff with the 2020 and 2026 Nobel Prizes in Economics | `step4_outputs.txt` |
| 5. Supervised fine-tuning | Three questions with their best-ranked human answers from the OpenAssistant dataset | `oasst_sample.json`, `conversations.json` |
| 6. Hallucinations, thinking and tools | Five questions about an obscure article (Jyderup) with and without the article, "Who created you?", the retailer profit calculation with thinking and non-thinking models, and letter/number tasks with and without code | `step6_outputs.txt` |
| 7. Run a model locally | Ran Qwen3 1.7B in LM Studio on my Mac and asked for a pelican joke | `local_model.png` |

## Main things I learned

- **Training data:** a model only knows what was in its training text. My tiny model could not even read "The sky is", because "sky" never appeared in its 10 documents.
- **Memorising vs. learning:** with so little data, my model copied sentences word for word and glued pieces of different texts together. That fluent but unchecked text is how hallucinations look.
- **Base model vs. assistant:** the base model continues a document; the instruct model answers. Post-training changes the behaviour, not the facts: both made confident factual errors.
- **Knowledge cutoff:** asked about the 2026 Nobel Prize, the base model invented two economists instead of saying it did not know.
- **Context and tools:** with the article in the context, or with web search, every answer about Jyderup was correct. From memory alone, the model admitted it did not know or gave rough answers.
- **Calculations:** the profit answers (€930.46) were correct because Gemini ran code. Asking a model to "use code" did not guarantee that it did: ChatGPT still got a letter task wrong.
- **Local models:** a model can run fully on my own computer, so company data never has to leave it.

Note: `requirements.txt`, `train.py`, `sky_test.py` and `corpus_test.py` were written by an LLM from the prompts in the assignment. I ran them myself, and all results are from my own runs.
