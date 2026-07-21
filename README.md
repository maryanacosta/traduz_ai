```markdown
# Traduz AI – Saúde em Evidência
> Abordagem Computacional para Ciência Aberta: Simplificação de Textos Científicos no Combate à Desinformação em Saúde

O Traduz AI é uma plataforma de Ciência Aberta baseada em Retrieval-Augmented Generation (Offline RAG) que transforma artigos científicos biomédicos em textos acessíveis para o público leigo, preservando o conteúdo científico por meio de validação semântica baseada em Deep Learning.

---

## Principais Funcionalidades

- Busca federada de artigos científicos
- Curadoria automática de evidências
- Simplificação textual em dois níveis
- Detecção e explicação de jargões
- Avaliação automática da fidelidade científica
- Portal web para divulgação científica

---

## Arquitetura do Sistema

           Mito em Saúde
                  │
                  ▼
      Busca Federada de Artigos
      (Semantic Scholar + Crossref)
                  │
                  ▼
        Curadoria Automática
                  │
                  ▼
     Painel CMS de Revisão (Streamlit)
                  │
                  ▼
      Pipeline de Simplificação
     ├── Diagnóstico Linguístico
     ├── Simplificação Nível 1
     ├── Simplificação Nível 2
     └── Validação Semântica
                  │
                  ▼
      Portal Público de Notícias

```

---

## Diferenciais Técnicos

| Componente | Descrição |
| --- | --- |
| Busca Federada | Integra Semantic Scholar e Crossref com deduplicação automática |
| Recuperação de PDFs | Estratégia em três camadas (Semantic Scholar -> Unpaywall -> OpenAlex) |
| ISR | Índice de Simplificação Real para avaliar ganhos de legibilidade |
| Validação Semântica | Similaridade vetorial utilizando MiniLM-L12 |
| Simplificação Multinível | Dois níveis adaptativos de simplificação textual |

---

## Tecnologias

| Categoria | Tecnologias |
| --- | --- |
| Linguagem | Python 3.10 / 3.11 |
| PLN | spaCy, Pyphen, deep-translator |
| IA Generativa | Gemini 2.5 Flash, Llama 3.3 |
| Embeddings | Sentence Transformers (MiniLM-L12) |
| Interface | Streamlit, HTML, CSS, JavaScript |
| APIs | Semantic Scholar, Crossref, OpenAlex |

---

## Estrutura do Projeto

```text
TraduzAI/
│
├── coletor_artigos.py
├── revisor.py
├── index.html
├── requirements.txt
├── noticias.json
├── artigos_coletados.json
└── .env

```

---

## Fluxo de Execução

```text
1️⃣ Coletor
    │
    ▼
artigos_coletados.json
    │
    ▼
2️⃣ Revisor (Streamlit)
    │
    ▼
noticias.json
    │
    ▼
3️⃣ Portal Web

```

---

## Instalação

### Instalação no Windows (PowerShell)

```powershell
cd "C:\caminho\para\o\seu\projeto"
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download pt_core_news_lg

```

### Instalação no Linux / WSL (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download pt_core_news_lg

```

### Configuração de Variáveis de Ambiente (`.env`)

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
USER_EMAIL=seu_email@ufv.br
SEMANTIC_SCHOLAR_KEY=sua_chave_semantic_scholar
GEMINI_API_KEY=sua_chave_gemini
GROQ_API_KEY=sua_chave_groq

```

---

## Executando

### 1. Coleta

```bash
python coletor_artigos.py

```

### 2. Revisão

```bash
streamlit run revisor.py

```

### 3. Portal

```bash
python -m http.server 8000

```

Acesse no navegador: `http://localhost:8000`

---

## Como Funciona a Simplificação

```text
Texto Científico
       │
       ▼
Diagnóstico Linguístico
       │
       ▼
Mapeamento de Jargões
       │
       ▼
Simplificação LLM
       │
       ▼
Validação MiniLM
       │
       ▼
Publicação

```

---

## Exemplo de Resultado

* **Entrada (Texto Científico Original):**
> "A fotoproteção reduz a incidência de carcinoma basocelular."


* **Nível 1 (Divulgação Popular):**
> "Usar protetor solar ajuda a diminuir o risco de alguns tipos de câncer de pele."


* **Nível 2 (Linguagem Simples - Acessibilidade Radical):**
> "Passar protetor solar ajuda a proteger a pele e pode evitar alguns tipos de câncer."



---

## Resultados Obtidos

| Indicador | Resultado |
| --- | --- |
| Artigos avaliados | 10 |
| Fidelidade científica | 100% |
| Similaridade média | 92,6% |
| Busca federada | Semantic Scholar + Crossref |
| Validação | MiniLM-L12 |

---

## Licença

Este projeto é distribuído sob a licença MIT. Para mais detalhes, consulte o arquivo de licença do repositório.

Autora: Maryana Costa do Vale (Universidade Federal de Viçosa)

```

```
