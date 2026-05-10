# AI / LLM Öğrenme Yol Haritası

> Tamamladıkça `[ ]` → `[x]` olarak işaretle.

---

## Aşama 1 — LLM API Temelleri

- [x] **OpenAI API ile ilk istek**
  Python veya JS ile `openai` kütüphanesini kur, basit bir chat completion gönder.

- [x] **Groq API ile ikinci istek** *(ücretsiz)*
  [console.groq.com](https://console.groq.com) üzerinden ücretsiz API key al. `groq` kütüphanesi ile Llama veya Mistral modeline istek gönder, OpenAI ile farkını karşılaştır.

- [x] **Function calling / tool use**
  Bir fonksiyon tanımla (örn. hava durumu), modelin bunu nasıl çağırdığını gözlemle.

- [x] **Prompt engineering pratiği**
  System prompt, few-shot örnekleri, chain-of-thought teknikleri üzerine 3 farklı senaryo yaz.

---

## Aşama 2 — Vektör Veritabanları

- [x] **Embedding nedir, nasıl üretilir?**
  `text-embedding-3-small` veya `all-MiniLM` ile birkaç cümleyi embed et, benzerliği hesapla.

- [x] **Qdrant'ı lokal kur**
  Docker ile Qdrant başlat, bir koleksiyon oluştur, vektör ekle ve sorgula.

- [x] **pgvector dene**
  PostgreSQL + pgvector ile basit bir vektör tablosu oluştur, `<->` operatörü ile sorgula.

- [x] **Pinecone veya Weaviate ile cloud deneyimi**
  Ücretsiz tier'da bir index/collection oluştur, 100+ vektör yükle, nearest-neighbor sorgula.

---

## Aşama 3 — RAG Mimarisi

- [x] **Naive RAG kur**
  Bir PDF'i chunk'la, embed et, vektör DB'ye yükle. Sorgu → retrieval → LLM cevabı pipeline'ı yaz.

- [x] **LangChain ile RAG**
  `RetrievalQA` veya LCEL kullanarak aynı pipeline'ı LangChain ile yeniden yap.

- [x] **Reranking ekle**
  Cohere Rerank veya cross-encoder ile retrieval kalitesini artır, farkı ölç.

- [x] **Hibrit arama dene**
  BM25 (keyword) + dense vector arama kombinasyonunu uygula.

---

## Aşama 4 — Agent Tabanlı Workflow

- [ ] **LangChain Agent kur**
  Web search + calculator tool'ları olan basit bir ReAct agent yaz.

- [ ] **LangGraph ile stateful agent**
  Bir graph tanımla, düğümler arası state akışını ve döngüyü anla.

- [ ] **AutoGen ile multi-agent**
  İki ajan oluştur (AssistantAgent + UserProxyAgent), birbirleriyle konuşturan bir task çöz.

- [ ] **Semantic Kernel dene**
  .NET veya Python'da SK ile bir "skill" ve "planner" oluştur.

---

## Aşama 5 — MCP Sunucu / İstemci

- [ ] **MCP'yi anla**
  Model Context Protocol spec'ini oku, tool/resource/prompt kavramlarını öğren.

- [ ] **Basit bir MCP sunucusu yaz**
  `@modelcontextprotocol/sdk` (JS) veya `mcp` (Python) ile tek tool'lu bir sunucu yaz.

- [ ] **MCP istemcisi yaz**
  Kendi sunucuna bağlanan bir istemci yaz, tool çağrısını manuel tetikle.

- [ ] **Claude Desktop ile entegrasyon**
  Yazdığın MCP sunucusunu Claude Desktop config'ine ekle, Claude'un tool'unu kullandığını gözlemle.

---

*Toplam: 20 görev — başarılar!*