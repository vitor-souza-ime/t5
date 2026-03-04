from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ===============================
# 1️⃣ Carregando o modelo T5
# ===============================

model_name = "google/flan-t5-base"

print("Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Modelo carregado em: {device}")
print("-" * 60)


# ===============================
# 2️⃣ Função geral texto → texto
# ===============================

def run_t5(prompt, max_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_tokens,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ===============================
# 3️⃣ Demonstração prática
# ===============================

examples = {
    "Resumo": "summarize: Artificial Intelligence is transforming education, industry and research by enabling automation and intelligent systems.",
    
    "Tradução": "translate English to Portuguese: Machine learning is fascinating.",
    
    "Pergunta e Resposta": "question: What is the capital of France? context: France is a country in Europe. Its capital is Paris.",
    
    "Classificação": "Classify sentiment: I absolutely loved this movie!"
}

for task, text in examples.items():
    print(f"\n🧠 Tarefa: {task}")
    print(f"Entrada: {text}")
    result = run_t5(text)
    print(f"Saída: {result}")
    print("-" * 60)


# ===============================
# 4️⃣ Explicação Final
# ===============================

print("\nConclusão:")
print("O T5 transforma TODAS as tarefas em um único formato: TEXTO → TEXTO.")
print("Mudamos apenas a instrução inicial do prompt.")
