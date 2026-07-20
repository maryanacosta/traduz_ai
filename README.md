# Traduz Ai - Saude em Evidencia
> Abordagem Computacional para Ciencia Aberta: Simplificacao de Textos Cientificos no Combate a Desinformacao em Saude

## Sobre o Projeto

O Traduz Ai e uma plataforma baseada em arquitetura de Geracao Aumentada por Recuperacao com Ingestao Previa (Offline Ingestion RAG). O sistema atua como um Jornal Cientifico Curatorial para Leigos, transformando artigos biomedicos complexos de alta evidencia clinica em narrativas inteligiveis e acessiveis para o publico leigo e individuos acometidos pelo analfabetismo funcional.

---

## Diferenciais Tecnicos

- Busca Federada Simultanea: Orquestracao de buscas paralelas e desduplicacao atomica via APIs do Semantic Scholar e Crossref.
- Resolucao de PDFs em 3 Camadas: Algoritmo de contingencia encadeado (Semantic Scholar -> Unpaywall -> OpenAlex).
- Metrica Delta Adaptativa (ISR): Funcao de score composto que avalia o ganho real de legibilidade textual (reducao de jargoes via IDF, simplificacao sintatica e enriquecimento explicativo).
- Modulo Antialucinacao via Deep Learning: Avaliacao vetorial cruzada de distancia semantica utilizando o modelo paraphrase-multilingual-MiniLM-L12-v2.
- Simplificacao Multinivel: Geracao adaptativa concorrente para Nivel 1 (Divulgacao Popular com glossario inline) e Nivel 2 (Linguagem Simples com ordem direta).

---

## Tecnologias Utilizadas

- Linguagem Principal: Python 3.10 / 3.11
- Processamento de Linguagem Natural (PLN): spaCy (modelo pt_core_news_lg), Pyphen, deep-translator
- Deep Learning & Embeddings: torch, torchvision, transformers, sentence-transformers, scikit-learn
- Orquestracao de LLMs: Google GenAI API (gemini-2.5-flash) / Groq API (llama-3.3-70b-versatile)
- Painel CMS & Interfaces: Streamlit (Revisor), HTML5, CSS3, JavaScript Nativo (Portal do Leitor)

---

## 1. Configuracao do Ambiente e Instalacao

### No Windows (PowerShell)

1. Abra o terminal na pasta do projeto:
   cd "C:\caminho\para\o\seu\projeto"

2. Crie o ambiente virtual com o Python 3.10 ou 3.11:
   py -3.10 -m venv .venv

3. Libere a execucao de scripts e ative a .venv:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1

4. Instale todas as dependencias:
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

5. Baixe o modelo em portugues do spaCy:
   python -m spacy download pt_core_news_lg

---

### No Linux / WSL (Ubuntu)

1. Navegue ate o diretorio e crie o ambiente virtual:
   python3 -m venv .venv

2. Ative a .venv:
   source .venv/bin/activate

3. Instale as dependencias e o modelo do spaCy:
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download pt_core_news_lg

---

## 2. Configuracao das Chaves de API (.env)

Crie um arquivo .env na raiz do projeto contendo as credenciais de acesso:

USER_EMAIL=seu_email@ufv.br
SEMANTIC_SCHOLAR_KEY=sua_chave_semantic_scholar
GEMINI_API_KEY=sua_chave_gemini
GROQ_API_KEY=sua_chave_groq

---

## 3. Como Rodar a Aplicacao

O ecossistema funciona atraves do fluxo encadeado de 3 etapas:

[Passo 1: Coletor] --> gera artigos_coletados.json
         |
[Passo 2: Revisor (CMS)] --> gera noticias.json (apos aprovacao)
         |
[Passo 3: Portal Publico] --> exibe index.html no navegador

### Passo 1 - Coleta Federada de Artigos
Busca simultaneamente nas APIs do Semantic Scholar e Crossref, unificando e ranqueando os estudos pelo score de evidencia clinica (Sf).

python coletor_artigos.py

Parametros opcionais:
- --por-tema 8 (Quantidade de artigos por tema - Padrao: 5)
- --min-score 30 (Score minimo de qualidade de evidencia)
- --temas "cancer" "nutrition" (Lista de temas especificos)

Saida: artigos_coletados.json

---

### Passo 2 - Painel CMS de Revisao e Simplificacao (Streamlit)
Interface visual para revisao dos artigos. Permite editar manchetes em portugues, aprovar, despublicar do site ou reavaliar estudos. Aprovados passam pelo pipeline de PLN (ISR + LLM + MiniLM-L12).

streamlit run revisor.py

O navegador abrira automaticamente em http://localhost:8501.
Saida: noticias.json

---

### Passo 3 - Portal de Noticias Publico (index.html)
Interface do leitor publico com suporte a busca, navegacao por temas, leitor de materias em tela cheia, abas por nivel de leitura, jargoes mapeados e audio por sintese de voz.

Para evitar bloqueios de seguranca do navegador (CORS), inicie o servidor HTTP local:

python -m http.server 8000

Acesse no navegador:
http://localhost:8000

---

## Estrutura de Arquivos Principais

.env                  # Variaveis de ambiente e chaves de API (nao versionado)
requirements.txt      # Dependencias completas do projeto
coletor_artigos.py    # Busca federada (Semantic Scholar + Crossref)
revisor.py            # Painel CMS em Streamlit (PLN + Simplificacao RAG)
index.html            # Portal web do leitor publico (Full-screen Reader)
artigos_coletados.json# Saida bruta da busca por evidencias
noticias.json         # Banco de noticias e simplificacoes publicadas
