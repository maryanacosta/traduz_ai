# -*- coding: utf-8 -*-
"""
Plataforma RAG de Acessibilidade Textual Científica - Mapeamento de Métricas
Nome do Projeto: Me Explica Aqui 💡
Versão: Otimização de Diagnóstico Híbrido com Resolução de PDFs em Camadas (Unpaywall + OpenAlex)
"""

import os
import sys
import time
import math
import re
import json
import logging
import tempfile
import hashlib
from functools import lru_cache
from typing import List, Dict, Set, Optional, Tuple
from dotenv import load_dotenv

# Configuração de Logging Estruturado para o TCC
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import requests
import pandas as pd
import spacy
import pyphen
import torch
from deep_translator import GoogleTranslator
from transformers import T5ForConditionalGeneration, AutoTokenizer
from analisador_semantico import SemanticDistanceAnalyzer

# ================================================================================
# VALIDAÇÃO DE AMBIENTE E CONFIGURAÇÃO DE SEGURANÇA
# ================================================================================

# Certifique-se de preencher essas chaves no seu arquivo .env
SEMANTIC_SCHOLAR_KEY = os.environ.get('SEMANTIC_SCHOLAR_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not GEMINI_API_KEY and not GROQ_API_KEY:
    logging.error("Nenhuma chave de API de LLM (GEMINI_API_KEY ou GROQ_API_KEY) foi detectada no ambiente.")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_LIB = True
except ImportError:
    HAS_GEMINI_LIB = False

try:
    from groq import Groq
    HAS_GROQ_LIB = True
except ImportError:
    HAS_GROQ_LIB = False

# ================================================================================
# MÓDULO NLP (MAPEAMENTO DE QUERIES COM PALAVRAS-CHAVE MANDATÓRIAS)
# ================================================================================

class NLPQueryMapper:
    def __init__(self):
        self.dicionario_tcc = {
            "A vacina contra COVID-19 altera o DNA humano": "vaccine COVID-19",
            "Vermicida cura o câncer de forma rápida": "anthelmintic cancer", 
            "Protetor solar causa câncer de pele": "sunscreen cancer",
            "Cloroquina cura a COVID-19": "cloroquina COVID-19",
            "A dieta da selva baseada apenas em carne e proteína é saudável e cura doenças": "meat protein diet health",
            "Água com açúcar acalma os nervos": "sugar water anxiety",
            "O ovo aumenta o colesterol e faz mal para o coração?": "egg cholesterol.",
            "Tomar 'friagem' ou sair de cabelo molhado causa dor de garganta": "Cold weather the flu."
        }

    def mapear_mito(self, mito: str) -> str:
        return self.dicionario_tcc.get(mito, re.sub(r'[^\w\s]', '', mito).lower())

# ================================================================================
# MOTOR DE BUSCA SIMULTÂNEO (FEDERATED SEARCH ENGINE)
# ================================================================================

class CompetitiveSearchEngine:
    def __init__(self):
        self.s2_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.crossref_url = "https://api.crossref.org/works"
        self.s2_key = SEMANTIC_SCHOLAR_KEY
        self.user_email = "maryana.vale@ufv.br" 

    def _buscar_semantic_scholar(self, query: str, limite: int) -> List[Dict]:
        logging.info("[Search] Disparando busca activa no Semantic Scholar...")
        params = {
            "query": query,
            "limit": limite,
            "fields": "paperId,title,abstract,publicationTypes,citationCount,openAccessPdf,year,authors,isRetracted"
        }
        headers = {"User-Agent": "Academic-TCC-Research-Bot/2.0", "Accept": "application/json"}
        if self.s2_key:
            headers["x-api-key"] = self.s2_key

        try:
            response = requests.get(self.s2_url, params=params, headers=headers, timeout=12)
            if response.status_code == 200:
                papers = response.json().get("data", [])
                logging.info(f"[Semantic Scholar] Encontrou {len(papers)} artigos candidatos.")
                return papers
        except Exception as e:
            logging.warning(f"[Semantic Scholar] Erro na busca: {e}")
        return []

    def _buscar_crossref(self, query: str, limite: int) -> List[Dict]:
        logging.info("[Search] Disparando busca activa na Crossref API...")
        params = {
            "query": query,
            "rows": limite,
            "select": "DOI,title,abstract,type,is-referenced-by-count,published,author"
        }
        headers = {"User-Agent": f"MeExplicaAqui-TCC/2.0 (mailto:{self.user_email})"}

        try:
            response = requests.get(self.crossref_url, params=params, headers=headers, timeout=12)
            if response.status_code == 200:
                items = response.json().get("message", {}).get("items", [])
                artigos_convertidos = []
                
                for item in items:
                    abstract = item.get("abstract", "")
                    if abstract:
                        abstract = re.sub(r'<[^>]+>', '', abstract).strip()
                    
                    if not abstract:
                        continue

                    titulo_lista = item.get("title", ["Sem Título"])
                    titulo = titulo_lista[0] if titulo_lista else "Sem Título"
                    
                    autores_brutos = item.get("author", [])
                    lista_autores = [{"name": f"{a.get('given', '')} {a.get('family', '')}".strip()} for a in autores_brutos]

                    ano = 2024
                    if "published" in item and "date-parts" in item["published"]:
                        parts = item["published"]["date-parts"]
                        if parts and parts[0]:
                            ano = parts[0][0]

                    # Correção: Salva o DOI real como paperId se disponível para viabilizar Unpaywall/OpenAlex
                    artigos_convertidos.append({
                        "paperId": item.get("DOI") if item.get("DOI") else f"crossref-{hash(titulo)}",
                        "title": titulo,
                        "abstract": abstract,
                        "publicationTypes": [item.get("type", "journal-article")],
                        "citationCount": item.get("is-referenced-by-count", 0),
                        "openAccessPdf": None,
                        "year": ano,
                        "authors": lista_autores,
                        "isRetracted": False
                    })
                logging.info(f"[Crossref] Encontrou {len(artigos_convertidos)} artigos candidatos com abstracts.")
                return artigos_convertidos
        except Exception as e:
            logging.warning(f"[Crossref] Erro na busca: {e}")
        return []

    def buscar_mistura_coesa(self, query: str, limite_por_fonte: int = 40) -> List[Dict]:
        resultados_s2 = self._buscar_semantic_scholar(query, limite_por_fonte)
        resultados_cr = self._buscar_crossref(query, limite_por_fonte)
        
        todos_artigos = []
        ids_vistos = set()
        
        for art in (resultados_s2 + resultados_cr):
            p_id = art.get("paperId")
            if p_id and p_id not in ids_vistos:
                ids_vistos.add(p_id)
                todos_artigos.append(art)
                
        logging.info(f"[Pool de Busca] Totalizado {len(todos_artigos)} artigos únicos reunidos para filtragem.")
        return todos_artigos

# ================================================================================
# MÓDULO CURADOR (FILTRAGEM INTERSEÇÃO TOTAL + RESOLUÇÃO DE PDFS EM CAMADAS)
# ================================================================================

class ArticleCurator:
    def __init__(self):
        self.TIPOS_ALTA_EVIDENCIA = {"Review", "SystematicReview", "ClinicalTrial", "MetaAnalysis", "journal-article"}
        self.user_email = "maryana.vale@ufv.br"

    def _resolver_url_pdf(self, paper: Dict) -> Optional[str]:
        """[PARTE 1] Tenta obter URL de PDF por múltiplas fontes, em ordem de confiabilidade."""
        # 1. PDF direto do Semantic Scholar (se disponível)
        pdf_info = paper.get("openAccessPdf")
        if pdf_info and pdf_info.get("url"):
            return pdf_info["url"]
        
        # 2. Unpaywall via DOI (gratuito, estável e oficial)
        doi = paper.get("paperId", "")
        if doi and doi.startswith("10."):  # DOIs válidos começam com "10."
            try:
                url = f"https://api.unpaywall.org/v2/{doi}?email={self.user_email}"
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    best = data.get("best_oa_location")
                    if best and best.get("url_for_pdf"):
                        return best["url_for_pdf"]
            except Exception:
                pass
        
        # 3. OpenAlex como Fallback estrutural
        if doi and doi.startswith("10."):
            try:
                url = f"https://api.openalex.org/works/doi:{doi}"
                resp = requests.get(url, timeout=8, 
                                    headers={"User-Agent": f"MeExplicaAqui/2.0 (mailto:{self.user_email})"})
                if resp.status_code == 200:
                    data = resp.json()
                    oa = data.get("open_access", {})
                    if oa.get("oa_url"):
                        return oa["oa_url"]
            except Exception:
                pass
        
        return None

    def avaliar_e_ranquear(self, artigos: List[Dict], query_origem: str) -> pd.DataFrame:
        artigos_processados = []
        ano_atual = 2026
        
        palavras_obrigatorias = [palavra.strip().lower() for palavra in query_origem.split() if palavra.strip()]
        logging.info(f"[Filtro Estrito] Exigindo a coexistência de TODAS as palavras: {palavras_obrigatorias}")

        for art in artigos:
            if art.get("isRetracted") or not art.get("abstract"):
                continue

            titulo = str(art.get("title", "")).lower()
            abstract = str(art.get("abstract", "")).lower()
            texto_completo = f"{titulo} {abstract}"

            passou_no_filtro_estrito = True
            for palavra in palavras_obrigatorias:
                if palavra not in texto_completo:
                    passou_no_filtro_estrito = False
                    break
            
            if not passou_no_filtro_estrito:
                continue

            tipos = art.get("publicationTypes") or []
            tipos_set = set(tipos)

            tem_alta_evidenca = len(tipos_set.intersection(self.TIPOS_ALTA_EVIDENCIA)) > 0
            base_score = 60.0 if tem_alta_evidenca else 20.0
            
            citations = art.get("citationCount", 0)
            log_citations_score = math.log1p(citations) * 8.0 

            ano = art.get("year") or (ano_atual - 5)
            idade_artigo = max(0, ano_atual - ano)
            decaimento_temporal = - (idade_artigo * 1.5)

            score_final = base_score + log_citations_score + decaimento_temporal

            # RESOLUÇÃO INTELIGENTE DE PDF (Substituição dinâmica pela nova arquitetura)
            pdf_url = self._resolver_url_pdf(art)

            lista_autores = art.get("authors") or []
            nomes_autores = [autor.get("name") for autor in lista_autores if autor.get("name")]
            string_autores = ", ".join(nomes_autores) if nomes_autores else "Autor Desconhecido"

            artigos_processados.append({
                "paperid": art.get("paperId", ""),
                "Título": art.get("title"),
                "Ano": ano,
                "Autores": string_autores,
                "Tipos": ", ".join(tipos) if tipos else "Artigo de Periódico (Journal)",
                "Citações": citations,
                "score_relevancia": score_final,
                "url_pdf": pdf_url,  # Persiste a URL do PDF resolvida em camadas
                "Abstract": art.get("abstract")
            })

        df = pd.DataFrame(artigos_processados)
        if not df.empty:
            df = df.sort_values(by="score_relevancia", ascending=False).reset_index(drop=True)
        return df

# ================================================================================
# MÓDULO DIAGNÓSTICO AVANÇADO (MÉTRICAS MULTIDIMENSIONAIS DELTA)
# ================================================================================

class AdvancedTextDiagnostic:
    def __init__(self):
        try:
            self.nlp = spacy.load("pt_core_news_lg")
        except OSError:
            raise RuntimeError("Execute antes: python -m spacy download pt_core_news_lg")
            
        self.dic_pt = pyphen.Pyphen(lang='pt_BR')
        self.conectores_explicativos = [
            "na verdade", "isso significa que", "ou seja", "por exemplo", 
            "isso acontece porque", "dessa forma", "para você entender", "sabe"
        ]
        self.lista_branca_jargoes = {
            "paciente", "pacientes", "saúde", "estudo", "pesquisa", "tratamento", "tratamentos",
            "médico", "médicos", "artigo", "evidência", "evidências", "caso", "casos", "efeito", "efeitos"
        }

    @lru_cache(maxsize=4096)
    def _contar_silabas(self, palavra: str) -> int:
        palavra_limpa = re.sub(r'[^\w]', '', palavra.lower())
        if not palavra_limpa:
            return 0
        hifenizada = self.dic_pt.inserted(palavra_limpa)
        return len(hifenizada.split('-')) if hifenizada else 1

    def calcular_idf_do_corpus(self, documentos: List[str]) -> Dict[str, float]:
        total_docs = len(documentos)
        if total_docs == 0:
            return {}
        palavras_por_doc = []
        vocabulario: Set[str] = set()

        for doc_texto in documentos:
            spacy_doc = self.nlp(doc_texto.lower())
            lemas_validos = {token.lemma_ for token in spacy_doc if token.is_alpha and not token.is_stop}
            palavras_por_doc.append(lemas_validos)
            vocabulario.update(lemas_validos)

        idf_dict = {}
        for termo in vocabulario:
            df_termo = sum(1 for doc_lemas in palavras_por_doc if termo in doc_lemas)
            idf_dict[termo] = math.log(total_docs / df_termo) + 1.0
        return idf_dict

    def _tamanho_medio_frases(self, doc) -> float:
        frases = list(doc.sents)
        total_frases = max(len(frases), 1)
        total_palavras = len([token for token in doc if token.is_alpha])
        return total_palavras / total_frases

    def extrair_jargoes(self, doc, dicionario_idf: Dict[str, float], threshold_idf: float = 1.3) -> List[str]:
        jargoes = []
        for token in doc:
            if not token.is_stop and token.is_alpha and token.pos_ in {"NOUN", "ADJ"}:
                texto_lower = token.text.lower()
                if texto_lower in self.lista_branca_jargoes:
                    continue
                lema = token.lemma_.lower()
                if self._contar_silabas(texto_lower) > 3 or dicionario_idf.get(lema, 0.0) >= threshold_idf:
                    jargoes.append(texto_lower)
        return list(set(jargoes))

    def _contar_jargoes_tecnicos(self, doc, dicionario_idf: Dict[str, float], threshold_idf: float = 1.3) -> int:
        return len(self.extrair_jargoes(doc, dicionario_idf, threshold_idf))

    def _contar_conectores_explicativos(self, texto: str) -> int:
        texto_min = texto.lower()
        total = 0
        for conector in self.conectores_explicativos:
            total += len(re.findall(r'\b' + re.escape(conector) + r'\b', texto_min))
        return total

    def _detectar_analogias(self, texto: str) -> List[str]:
        padroes = [
            r'\b(?:funciona\s+como|são\s+como|é\s+como|parece\s+com)\s+([^.,;!?]{3,30})',
            r'\bcomo\s+(?:um|uma|se\s+fosse)\s+([^.,;!?]{3,30})'
        ]
        analogias = []
        for padrao in padroes:
            encontrados = re.findall(padrao, texto, re.IGNORECASE)
            for item in encontrados:
                item_clean = item.strip()
                if item_clean:
                    analise_item = self.nlp(item_clean)
                    possui_conteudo = any(t.pos_ in {"NOUN", "VERB", "ADJ"} for t in analise_item)
                    if possui_conteudo:
                        analogias.append(item_clean)
        return list(set(analogias))

    def calcular_indice_simplificacao_real(self, texto_original: str, texto_simplificado: str, dicionario_idf: Dict[str, float], nivel: str = "leve") -> float:
        texto_simp_limpo = re.sub(r'\([^)]*\)', '', texto_simplificado)
        texto_simp_limpo = re.sub(r'\s+', ' ', texto_simp_limpo).strip()

        doc_orig = self.nlp(texto_original)
        doc_simp = self.nlp(texto_simp_limpo)
        
        jargoes_orig = self._contar_jargoes_tecnicos(doc_orig, dicionario_idf)
        jargoes_simp = self._contar_jargoes_tecnicos(doc_simp, dicionario_idf)
        reducao_jargoes = max(0.0, (jargoes_orig - jargoes_simp) / max(jargoes_orig, 1)) * 100.0
        
        tamanho_frase_orig = self._tamanho_medio_frases(doc_orig)
        tamanho_frase_simp = self._tamanho_medio_frases(doc_simp)
        reducao_sintaxe = max(0.0, (tamanho_frase_orig - tamanho_frase_simp) / max(tamanho_frase_orig, 1.0)) * 100.0
        
        conectores_orig = self._contar_conectores_explicativos(texto_original)
        conectores_simp = self._contar_conectores_explicativos(texto_simplificado)
        melhoria_explicativa = max(0.0, (conectores_simp - conectores_orig)) * 10.0
        
        analogias = self._detectar_analogias(texto_simplificado)
        bonus_analogias = len(analogias) * 5.0
        
        if nivel == "forte":
            pesos = {"jargao": 0.35, "sintaxe": 0.45, "explicativo": 0.10, "analogia": 0.10}
        else:
            pesos = {"jargao": 0.40, "sintaxe": 0.25, "explicativo": 0.25, "analogia": 0.10}

        indice_simplificacao = (
            (reducao_jargoes * pesos["jargao"]) + 
            (reducao_sintaxe * pesos["sintaxe"]) + 
            (melhoria_explicativa * pesos["explicativo"]) + 
            (bonus_analogias * pesos["analogia"])
        )
        return round(max(0.0, min(100.0, indice_simplificacao)), 2)

    def diagnosticar(self, texto: str, dicionario_idf: Dict[str, float], threshold_idf: float = 1.3) -> Dict:
        texto_limpo = re.sub(r'\([^)]*\)', '', texto)
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()

        doc = self.nlp(texto_limpo)
        frases = list(doc.sents)
        total_frases = max(len(frases), 1)
        palavras = [token for token in doc if token.is_alpha]
        total_palavras = len(palavras)

        if total_palavras == 0:
            return {"ilf": 0.0}

        total_silabas = sum(self._contar_silabas(token.text) for token in palavras)
        msl = total_palavras / total_frases
        asw = total_silabas / total_palavras

        ilf_tradicional = 248.83 - (1.015 * msl) - (84.6 * asw)
        total_jargoes = self._contar_jargoes_tecnicos(doc, dicionario_idf, threshold_idf)
        densidade_jargoes = total_jargoes / total_palavras
        v_ease = (1.0 - densidade_jargoes) * 100.0

        total_verbos = sum(1 for token in doc if token.pos_ in {"VERB", "AUX"})
        total_nominais = sum(1 for token in doc if token.pos_ in {"NOUN", "ADJ"})
        divisor_nominal = max(total_nominais + total_verbos, 1)
        proporcao_verbos = total_verbos / divisor_nominal
        s_verb = min(proporcao_verbos * 3.0, 1.0) * 100.0

        ihfc = (0.4 * ilf_tradicional) + (0.4 * v_ease) + (0.2 * s_verb)
        return {"ilf": max(0.0, min(100.0, round(ihfc, 2)))}

# ================================================================================
# MÓDULO UNIVERSAL DE SIMPLIFICAÇÃO TEXTUAL 
# ================================================================================

class HuggingFaceSimplifier:
    def __init__(self, model_name="unicamp-dl/ptt5-base-portuguese-vocab"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        except Exception:
            self.model = None

    def simplificar_estrutura(self, texto: str, max_length: int = 256) -> str:
        if not texto or self.model is None:
            return texto
        try:
            inputs = self.tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception:
            return texto

class LLMFluidRefiner:
    def __init__(self):
        self.gemini_client = None
        self.groq_client = None

        if HAS_GEMINI_LIB and GEMINI_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception:
                pass
        if HAS_GROQ_LIB and GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
            except Exception:
                pass

    def _criar_prompt_base_pedagogico(self) -> str:
        return (
            "Considere as seguintes instruções de controle linguístico:\n"
            "- Substitua termos do português europeu por equivalentes brasileiros (ex: 'cancro' -> 'câncer').\n"
            "- Use conectores fluidos como 'na verdade', 'isso acontece porque', 'por exemplo'.\n"
            "- Mantenha tom acolhedor, educativo e estritamente focado em divulgação científica popular.\n"
            "- PROIBIÇÃO CRÍTICA: Não invente dados estatísticos, percentuais, nem adicione referências a instituições como 'OMS' ou 'Anvisa'.\n\n"
        )

    def _criar_prompt_explicacao_termos(self, jargoes_detectados: List[str]) -> str:
        if not jargoes_detectados:
            return ""
        return f"🔬 PALAVRAS COMPLEXAS IDENTIFICADAS NO TEXTO: [{', '.join(jargoes_detectados)}]\n"

    def _criar_prompt_nivel_leve(self, mito: str, jargoes_detectados: List[str]) -> str:
        return self._criar_prompt_base_pedagogico() + self._criar_prompt_explicacao_termos(jargoes_detectados) + f"🎯 Explique por que o mito '{mito}' é incorreto. Escreva em um parágrafo contínuo, sem marcações Markdown, com início, meio e fim estruturados."

    def _criar_prompt_nivel_forte(self, mito: str, jargoes_detectados: List[str]) -> str:
        return self._criar_prompt_base_pedagogico() + self._criar_prompt_explicacao_termos(jargoes_detectados) + f"🎯 ACESSIBILIDADE RADICAL: Explique por que o mito '{mito}' é falso. Use frases curtíssimas (8-10 palavras), ordem direta, retire porcentagens/anos, em um único parágrafo fluido sem asteriscos."

    def refinar_e_informar(self, texto_artigo: str, mito: str, nivel: str = "leve", jargoes_obrigatorios: list = None) -> str:
        if not texto_artigo:
            return "Texto indisponível."
        jargoes_obrigatorios = jargoes_obrigatorios or []
        prompt_sistema = self._criar_prompt_nivel_forte(mito, jargoes_obrigatorios) if nivel == "forte" else self._criar_prompt_nivel_leve(mito, jargoes_obrigatorios)
        conteudo_usuario = f"TEXTO:\n{texto_artigo}"

        if self.gemini_client:
            try:
                config_gemini = types.GenerateContentConfig(system_instruction=prompt_sistema, temperature=0.0)
                response = self.gemini_client.models.generate_content(model='gemini-2.5-flash', contents=conteudo_usuario, config=config_gemini)
                return response.text.strip()
            except Exception:
                pass
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": prompt_sistema}, {"role": "user", "content": conteudo_usuario}],
                    temperature=0.0
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                return "[Falha temporária de infraestrutura de IA]"
        return "Chaves de API ausentes."

class TranslationProvider:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='pt')
        self.cache = {}

    def traduzir_texto(self, texto: str) -> str:
        if not texto:
            return ""
        hash_key = hashlib.md5(texto.encode('utf-8')).hexdigest()
        if hash_key in self.cache:
            return self.cache[hash_key]
        try:
            traducao = self.translator.translate(texto)
            self.cache[hash_key] = traducao
            return traducao
        except Exception:
            return texto

# ================================================================================
# EXPORTAÇÃO COMPLETA DE DADOS EM CAMADAS (JSON)
# ================================================================================

def salvar_no_banco_json(mito_nome: str, dados_artigos: List[Dict]):
    arquivo_json = 'mitos.json'
    base_mitos = []
    
    if os.path.exists(arquivo_json) and os.path.getsize(arquivo_json) > 0:
        try:
            with open(arquivo_json, 'r', encoding='utf-8') as f:
                base_mitos = json.load(f)
        except json.JSONDecodeError:
            pass

    mito_id = "".join(c for c in mito_nome.lower().replace(" ", "-") if c.isalnum() or c == "-")
    base_mitos = [item for item in base_mitos if item.get('id') != mito_id]

    novo_registro = {
        "id": mito_id,
        "titulo_mito": str(mito_nome).strip(),
        "artigos": []
    }

    for art in dados_artigos:
        # [PARTE 2] Mapeamento explícito de campos no JSON final
        novo_registro["artigos"].append({
            "posicao": int(art["posicao"]),
            "titulo": str(art["titulo"]).strip(),
            "titulo_pt": str(art["titulo_pt"]).strip(),
            "ano": int(art["ano"]),
            "autores": str(art["autores"]).strip(),
            "score_relevancia": float(art["score_relevancia"]),
            "ilf_original": float(art["ilf_original"]),
            "ilf_leve": float(art["ilf_leve"]),
            "ilf_forte": float(art["ilf_forte"]),
            "similaridade_original_leve": float(art.get("sim_origem_leve", 0.0)),
            "similaridade_original_forte": float(art.get("sim_origem_forte", 0.0)),
            "status_fidelidade": str(art.get("status_fidelidade", "Não Avaliado")).strip(),
            
            # Persistência separada e explícita dos dois canais de roteamento
            "url_artigo": str(art["url_artigo"]).strip(),
            "url_pdf": str(art["url_pdf"]).strip(),       # Armazena a string direta do PDF
            "tem_download_direto": bool(art["tem_download_direto"]),
            
            "leve": str(art.get("simplificado_leve", "")).strip(),
            "forte": str(art.get("simplificado_forte", "")).strip()
        })

    base_mitos.append(novo_registro)
    
    dir_name = os.path.dirname(os.path.abspath(arquivo_json)) or '.'
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
        json.dump(base_mitos, tf, indent=2, ensure_ascii=False)
        tempname = tf.name
    os.replace(tempname, arquivo_json)
    logging.info(f"[JSON] Sucesso! Banco atualizado para o mito: '{mito_nome}'")

# ================================================================================
# ORQUESTRADOR CENTRAL
# ================================================================================

if __name__ == "__main__":
    logging.info("="*80)
    logging.info("INICIANDO PIPELINE RAG ACADÊMICO EM CAMADAS")
    logging.info("="*80)

    mito_selecionado = "Vermicida cura o câncer de forma rápida"

    mapper = NLPQueryMapper()
    search_engine = CompetitiveSearchEngine() 
    curator = ArticleCurator()
    diagnosticador = AdvancedTextDiagnostic()
    hf_engine = HuggingFaceSimplifier()
    llm_engine = LLMFluidRefiner()
    tradutor_cached = TranslationProvider()

    query_tecnica = mapper.mapear_mito(mito_selecionado)
    artigos_brutos = search_engine.buscar_mistura_coesa(query_tecnica, limite_por_fonte=40)
    df_ranking = curator.avaliar_e_ranquear(artigos_brutos, query_origem=query_tecnica)

    if df_ranking.empty:
        logging.error("Nenhum artigo passou no filtro estrito.")
        sys.exit(1)

    df_ranking.columns = [col.lower() for col in df_ranking.columns]
    analisador_distancia = SemanticDistanceAnalyzer(usar_deep_learning=True)
    df_top5 = df_ranking.head(5)

    lista_abstracts = [str(row.get("abstract", "")) for _, row in df_top5.iterrows()]
    dicionario_idf_calculado = diagnosticador.calcular_idf_do_corpus(lista_abstracts)

    relatorio_lote = []

    for idx, row in df_top5.reset_index(drop=True).iterrows():
        titulo_atual = row.get("título", row.get("title", f"Artigo #{idx+1}"))
        abstract_bruto = row.get("abstract", "")
        autores_artigo = row.get("autores", "Autor não especificado")

        if not abstract_bruto or str(abstract_bruto).strip() == "":
            continue

        titulo_pt = tradutor_cached.traduzir_texto(titulo_atual)
        abstract_pt = tradutor_cached.traduzir_texto(abstract_bruto)
        abstract_pt = abstract_pt.replace("ADN", "DNA").replace("Adn", "Dna").replace("cancro", "câncer")

        # Roteamento seguro da página de metadados estática (Evita erro 404 e 504)
        paper_id = str(row.get("paperid", ""))
        if "/" in paper_id or paper_id.startswith("10."):
            url_artigo = f"https://doi.org/{paper_id}"
        else:
            url_artigo = f"https://www.semanticscholar.org/paper/{paper_id}"

        # Recupera o PDF resolvido pelo resolvedor multifonte do Curador
        url_pdf = row.get("url_pdf")
        url_pdf_final = str(url_pdf) if url_pdf and pd.notna(url_pdf) else ""

        diag_original = diagnosticador.diagnosticar(abstract_pt, dicionario_idf_calculado)
        
        doc_original_analise = diagnosticador.nlp(abstract_pt)
        jargoes_detectados = diagnosticador.extrair_jargoes(doc_original_analise, dicionario_idf_calculado)
        
        saida_leve = llm_engine.refinar_e_informar(abstract_pt, mito_selecionado, "leve", jargoes_detectados)
        saida_forte = llm_engine.refinar_e_informar(abstract_pt, mito_selecionado, "forte", jargoes_detectados)

        if "Chaves de API ausentes" in saida_leve or "[Falha" in saida_leve:
            saida_leve = hf_engine.simplificar_estrutura(abstract_pt)
        if "Chaves de API ausentes" in saida_forte or "[Falha" in saida_forte:
            saida_forte = hf_engine.simplificar_estrutura(abstract_pt)

        ilf_leve_real = diagnosticador.calcular_indice_simplificacao_real(abstract_pt, saida_leve, dicionario_idf_calculado, nivel="leve")
        ilf_forte_real = diagnosticador.calcular_indice_simplificacao_real(abstract_pt, saida_forte, dicionario_idf_calculado, nivel="forte")

        analise_semantica = analisador_distancia.analisar_alinhamento(texto_original=abstract_pt, texto_leve=saida_leve, texto_forte=saida_forte)

        # [PARTE 2] Salvando explicitamente no dicionário temporário
        relatorio_lote.append({
            "posicao": idx + 1,
            "titulo": titulo_atual,
            "titulo_pt": titulo_pt,
            "ano": int(row.get("ano", 2026)),
            "autores": autores_artigo,
            "score_relevancia": round(float(row.get("score_relevancia", 0.0)), 2),
            "ilf_original": diag_original["ilf"], 
            "ilf_leve": ilf_leve_real,   
            "ilf_forte": ilf_forte_real, 
            "sim_origem_leve": analise_semantica.get("similaridade_original_leve", 0.0),
            "sim_origem_forte": analise_semantica.get("similaridade_original_forte", 0.0),
            "status_fidelidade": analise_semantica.get("status_fidelidade", "Sem Análise"),
            
            "url_artigo": url_artigo,
            "url_pdf": url_pdf_final,
            "tem_download_direto": len(url_pdf_final) > 0,
            
            "simplificado_leve": saida_leve,
            "simplificado_forte": saida_forte
        })
        time.sleep(1.0)

    salvar_no_banco_json(mito_selecionado, relatorio_lote)