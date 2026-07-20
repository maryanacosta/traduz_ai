# 🔬 Traduz Ai · Saúde em Evidência
> **Abordagem Computacional para Ciência Aberta: Simplificação de Textos Científicos no Combate à Desinformação em Saúde**

## 📌 Sobre o Projeto

O **Traduz Ai** é uma plataforma baseada em arquitetura de Geração Aumentada por Recuperação com Ingestão Prévia (*Offline Ingestion RAG*). O sistema atua como um **Jornal Científico Curatorial para Leigos**, transformando artigos biomédicos complexos de alta evidência clínica em narrativas inteligíveis e acessíveis para o público leigo e indivíduos acometidos pelo analfabetismo funcional.

---

## ⚡ Diferenciais Técnicos

- **Busca Federada Simultânea:** Orquestração de buscas paralelas e desduplicação atômica via APIs do Semantic Scholar e Crossref.
- **Resolução de PDFs em 3 Camadas:** Algoritmo de contingência encadeado (*Semantic Scholar ➔ Unpaywall ➔ OpenAlex*).
- **Métrica Delta Adaptativa ($I_{SR}$):** Função de score composto que avalia o ganho real de legibilidade textual (redução de jargões via IDF, simplificação sintática e enriquecimento explicativo).
- **Módulo Antialucinação via Deep Learning:** Avaliação vetorial cruzada de distância semântica utilizando o modelo `paraphrase-multilingual-MiniLM-L12-v2`.
- **Simplificação Multinível:** Geração adaptativa concorrente para Nível 1 (Divulgação Popular com glossário *inline*) e Nível 2 (Linguagem Simples com ordem direta).

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem Principal:** Python 3.10 / 3.11
- **Processamento de Linguagem Natural (PLN):** `spaCy` (modelo `pt_core_news_lg`), `Pyphen`, `deep-translator`
- **Deep Learning & Embeddings:** `torch`, `torchvision`, `transformers`, `sentence-transformers`, `scikit-learn`
- **Orquestração de LLMs:** Google GenAI API (`gemini-2.5-flash`) / Groq API (`llama-3.3-70b-versatile`)
- **Painel CMS & Interfaces:** Streamlit (Revisor), HTML5, CSS3, JavaScript Nativo (Portal do Leitor)

---

## ⚙️ 1. Configuração do Ambiente e Instalação

### No Windows (PowerShell)

1. **Abra o terminal na pasta do projeto:**
   ```powershell
   cd "C:\caminho\para\o\seu\projeto"
