# Traduz Ai

> **Abordagem Computacional para Ciência Aberta: Simplificação de Textos Científicos no Combate à Desinformação em Saúde**
> Monografia apresentada ao curso de Ciência da Computação da Universidade Federal de Viçosa (UFV) como parte das exigências para aprovação em TCC I.

---

## Sobre o Projeto
O **Traduz Ai** é uma plataforma baseada em arquitetura Geração Aumentada por Recuperação com Ingestão Prévia (*Offline Ingestion RAG*). O sistema atua como um *Jornal Científico Curatorial para Leigos*, transformando artigos biomédicos complexos de alta evidência clínica em narrativas inteligíveis e acessíveis para o público leigo e indivíduos acometidos pelo analfabetismo funcional.

### Diferenciais Técnicos
* **Busca Federada Simultânea:** Orquestração de buscas paralelas via APIs do *Semantic Scholar* e *Crossref*.
* **Resolução de PDFs em 3 Camadas:** Algoritmo de contingência encadeado (*Semantic Scholar* ➔ *Unpaywall* ➔ *OpenAlex*).
* **Métrica Delta Adaptativa ($I_{SR}$):** Função de score composto que avalia o ganho real de legibilidade textual.
* **Módulo Antialucinação Deep Learning:** Avaliação vetorial cruzada de distância semântica utilizando o modelo `paraphrase-multilingual-MiniLM-L12-v2`.

---

## Tecnologias Utilizadas
* **Linguagem Principal:** Python 3.10+
* **Processamento de Linguagem Natural (PLN):** `spaCy` (modelo `pt_core_news_lg`), `Pyphen`
* **Deep Learning & Embeddings:** `transformers`, `torch`, `sentence-transformers`
* **Orquestração de LLMs:** Google GenAI API (Gemini 2.5-Flash) / Groq API (Llama 3.1)
* **Interface Visual:** HTML5, CSS3 (Paged Media) e Compilação Industrial via `WeasyPrint`

---
