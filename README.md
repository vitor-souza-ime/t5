# T5 – Demonstração do Paradigma Texto → Texto

Este repositório apresenta uma implementação prática do modelo **FLAN-T5** utilizando Python e a biblioteca Hugging Face Transformers.

O objetivo do projeto é demonstrar o princípio central da arquitetura **T5 (Text-to-Text Transfer Transformer)**:  
todas as tarefas de Processamento de Linguagem Natural podem ser formuladas como um problema unificado de **texto → texto**.

---

## 📚 Sobre o Modelo

O projeto utiliza o modelo:

- `google/flan-t5-base`

O FLAN-T5 é uma versão instruída do T5, treinada para compreender melhor comandos em linguagem natural (prompt-based learning).

Com ele, diferentes tarefas são resolvidas apenas alterando a instrução textual no prompt, sem modificar a arquitetura do modelo.

Exemplos de tarefas demonstradas:

- ✅ Resumo
- ✅ Tradução
- ✅ Pergunta e Resposta
- ✅ Classificação de Sentimento

---

## 🧠 Conceito Central

O T5 propõe que:

> Toda tarefa de NLP pode ser convertida em um problema de transformação de texto.

Em vez de usar modelos diferentes para cada tarefa, utilizamos:
- O **mesmo modelo**
- A **mesma função**
- Alterando apenas o texto de entrada

Exemplo:

```

summarize: texto...
translate English to Portuguese: texto...
Classify sentiment: texto...

````

---

## ⚙️ Requisitos

Instale as dependências:

```bash
pip install transformers sentencepiece torch
````

---

## 🚀 Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/vitor-souza-ime/t5.git
cd t5
```

2. Execute o script Python:

```bash
python nome_do_arquivo.py
```

Ou utilize o **Google Colab** para execução com GPU.

---

## 🏗 Estrutura do Código

O script está organizado em quatro partes principais:

### 1️⃣ Carregamento do Modelo

* Carrega o tokenizador e o modelo `google/flan-t5-base`
* Detecta automaticamente CPU ou GPU

### 2️⃣ Função Genérica Texto → Texto

Função `run_t5(prompt)`:

* Tokeniza o texto
* Executa inferência
* Gera saída utilizando beam search
* Decodifica o resultado

### 3️⃣ Demonstração Prática

Executa múltiplas tarefas usando o mesmo modelo:

* Resumo
* Tradução
* Pergunta e resposta
* Classificação

### 4️⃣ Conclusão

Reforça o princípio do paradigma unificado do T5.

---

## 🔍 Exemplo de Saída

```
🧠 Tarefa: Tradução
Entrada: translate English to Portuguese: Machine learning is fascinating.
Saída: O aprendizado de máquina é fascinante.
```

---

## 🏛 Fundamentos Técnicos

O T5 é baseado na arquitetura **Transformer encoder-decoder**, utilizando:

* Multi-Head Attention
* Camadas Feed-Forward
* Geração autoregressiva
* Beam Search para melhoria de qualidade

O modelo foi pré-treinado com objetivo de **denoising**, aprendendo a reconstruir trechos mascarados de texto.

---

## 🎯 Objetivo Educacional

Este projeto foi desenvolvido com finalidade didática para:

* Demonstrar o funcionamento prático do T5
* Ilustrar o conceito de aprendizado por transferência
* Mostrar como múltiplas tarefas podem ser resolvidas com um único modelo generativo

---

## 📖 Referências

* Raffel et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*
* Hugging Face Transformers Documentation

---

## 👨‍💻 Autor

Vitor Souza
