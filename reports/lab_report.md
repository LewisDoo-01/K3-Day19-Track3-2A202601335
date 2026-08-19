# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Đỗ Tuấn Kiệt - 2A202601335 
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

## PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*

- **Ví dụ từ dữ liệu:** `chunk_id = 2373d95259772acd2345::c0000`
  - Gốc: *"Systems Solution Inc. (SSI) a leading provider of IT solutions and consulting for business today announced that **they** are opening **their** new office in Wayne PA. After almost three decades in **their** King of Prussia headquarters..."*
  - Sau coref: *"...announced that **Systems Solution Inc. (SSI)** are opening **Systems Solution Inc. (SSI)** new office in Wayne PA. After almost three decades in **Systems Solution Inc. (SSI)** King of Prussia headquarters..."*
- **Hiện tượng:** Đối tượng được resolve là *đúng* (SSI đúng là "they"/"their"), nhưng cơ chế **thay thế sai cách**: LLM dán nguyên văn cả cụm `"Systems Solution Inc. (SSI)"` (bao gồm phần viết tắt trong ngoặc) vào mọi vị trí đại từ, kể cả sở hữu cách — tạo ra câu sai ngữ pháp (`"Systems Solution Inc. (SSI) new office"` thay vì `"Systems Solution Inc.'s new office"`) và lặp cụm dài 3 lần trong một đoạn ngắn.
- **Hậu quả đối với Graph:** Đây không phải False Edge trực tiếp, mà là rủi ro **Entity Fragmentation**: chuỗi `"Systems Solution Inc. (SSI)"` sau khi qua `norm_entity()` ở Module 3 sẽ khác với các mention `"Systems Solution Inc."` (không kèm `(SSI)`) xuất hiện ở các chunk khác trong dataset. Nếu độ tương đồng vector giữa hai dạng chuẩn hoá này rơi dưới ngưỡng `0.90`, Entity Resolution có thể tách chúng thành 2 `entity_id` khác nhau cho cùng một công ty, làm phân mảnh đồ thị và giảm degree/coverage thực của node đó khi truy vấn.
- **Thống kê tổng quan (400 chunk đưa qua coref):** 182 chunk có `unresolved_mentions` (giữ nguyên vì ambiguous, đúng theo conservative rule), 123 chunk bị thay đổi nội dung bởi coref — trong đó phần lớn resolve đúng đối tượng, nhưng case trên cho thấy "đúng đối tượng" không đồng nghĩa "an toàn cho pipeline downstream".

---

### 2. Entity Resolution Threshold & Lexical Guard

> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*

- **Ngưỡng cosine similarity:** `threshold = 0.90` (chỉ log vào audit khi score ≥ threshold), kết hợp Lexical Guard: khớp tuyệt đối sau khi bỏ hậu tố công ty (`strip_suffix`), hoặc `SequenceMatcher.ratio() ≥ 0.72`.
- **Kết quả thực nghiệm trên dữ liệu thật:** Với 24 thực thể trích xuất được từ 1500 bài báo mẫu (sau dedup), **không có cặp nào đạt similarity ≥ 0.85** — độ tương đồng cao nhất đo được chỉ là **0.3987** (`Mark Leary` vs `Richard Jamieson`, đều bị Lexical Guard từ chối đúng — không trùng cụm từ, ratio thấp). `entity_resolution_audit_df` vì vậy **trống (0 dòng)** ở threshold 0.90, không đạt yêu cầu "≥10 dòng audit" của đề bài.
- **Nguyên nhân:** Extraction dùng prompt "prefer precision over recall" trên mẫu 1500 bài/400 chunk ngẫu nhiên (`SEED=42`) chỉ trích xuất được rất ít quan hệ rõ ràng (~12 triples tích lũy qua nhiều lần chạy), và các công ty xuất hiện trong mẫu này (Akeneo, SITA, Eseye, Zanaris, GreenPages...) đều là các công ty nhỏ/khác biệt, không có biến thể tên như Microsoft/MSFT hay Apple/Apple Music — nên cơ chế Lexical Guard **không có cơ hội được kích hoạt thật** trong lần chạy này.
- **Minh hoạ đúng thiết kế (theo case đề bài chỉ định, không phải dữ liệu thực nghiệm):** Guard được thiết kế để chặn chính xác 2 loại lỗi mà `ASSIGNMENT.md` nêu — (1) người trùng họ: `Sam Altman` vs `Steve Altman` → embedding có thể cho similarity cao (cùng cấu trúc "Tên + họ" phổ biến) nhưng `strip_suffix` không đổi gì (không phải công ty) và `SequenceMatcher` trên 2 chuỗi khác `first name` sẽ cho ratio thấp hơn 0.72 → **REJECT** đúng; (2) sản phẩm mang tên công ty: `Apple` vs `Apple Watch` → sau `strip_suffix` hai chuỗi vẫn khác nhau (`"apple"` vs `"apple watch"`), ratio ~0.67–0.70 (dưới 0.72) → **REJECT** đúng, tránh gộp nhầm sản phẩm vào node công ty mẹ.
- **Kết luận:** Ngưỡng 0.90 + guard 0.72 là hợp lý và an toàn cho các case rủi ro cao, nhưng đề bài giả định một dataset lớn/đa dạng hơn để guard "có việc để làm" — với mẫu 1500 bài scale-guard của lab, hiện tượng này chưa xảy ra tự nhiên.

---

### 3. Đồ thị & Super-node Mitigation

> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*

- **Kết quả thực tế:** `graph_checks()` trên đồ thị thật (24 node, 12 edge, 0 cạnh thiếu provenance) cho thấy **mọi node đều có degree = 1** — không có super-node (degree > 100) nào xuất hiện.

| Hạng | Tên thực thể                        | Loại thực thể (Type) | Bậc kết nối (Degree) |
| ----- | -------------------------------------- | ----------------------- | ----------------------- |
| 1     | (không có node nào vượt degree=1) | —                      | 1                       |
| 2     | —                                     | —                      | 1                       |
| 3     | —                                     | —                      | 1                       |

- **Vì sao không quan sát được Super-node ở scale này:** (1) `EXTRACTION_MAX_CHUNKS=400` + prompt "prefer precision over recall" chỉ sinh ra ~12 triples tích lũy trên 24 entity riêng biệt — mỗi entity chỉ xuất hiện đúng 1 lần trong toàn bộ mẫu; (2) 1500 bài báo được `sample()` ngẫu nhiên (`SEED=42`) từ tập tin tức đa ngành, phần lớn là các công ty nhỏ/vừa chỉ xuất hiện 1 lần, không giống dữ liệu tin tức tập trung vào Big Tech (Google/Microsoft/Meta) như ví dụ minh hoạ trong ASSIGNMENT.md. Super-node chỉ thực sự xuất hiện khi scale lên hàng chục nghìn bài báo, nơi các công ty lớn được nhắc lặp lại xuyên suốt nhiều bài.
- **Ưu điểm & Rủi ro của Temporal Mitigation (đánh giá theo thiết kế, áp dụng khi scale lớn):**
  - *Ưu điểm:* Khi một node thật sự có degree > 100 (vd Google, Microsoft ở dataset đầy đủ), giới hạn 50 cạnh mới nhất theo `published_date DESC` giữ ngữ cảnh trả lời tập trung vào diễn biến gần đây, tránh `MAX_GRAPH_CONTEXT_CHARS=14000` bị tràn bởi hàng trăm sự kiện cũ, đồng thời giữ latency truy vấn ổn định.
  - *Rủi ro:* Câu hỏi dạng lịch sử ("Microsoft đã mua bao nhiêu công ty AI kể từ 2015?") sẽ bị cắt mất các sự kiện cũ hơn 50 cạnh gần nhất → GraphRAG trả lời thiếu hoặc sai số liệu tổng hợp dù dữ liệu vẫn tồn tại trong graph, chỉ là không được truy xuất. Đây là đánh đổi Recency vs Completeness có chủ đích của thiết kế.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

> **Lưu ý về tính lặp lại (reproducibility):** Do model extraction/generation (`gpt-oss` reasoning model qua Groq) vẫn còn phương sai giữa các lần gọi dù `temperature=0.0`, số liệu benchmark **khác nhau đáng kể giữa các lần chạy pipeline trong cùng một buổi** (đã quan sát trực tiếp: 2 lần chạy liên tiếp cho ra 2 bộ điểm Judge khác nhau trên cùng 5 câu hỏi). Bảng dưới đây là **số liệu của lần chạy cuối cùng** — trùng khớp với 2 file `outputs/graphrag_eval_results.csv` và `outputs/graphrag_vs_flatrag_summary.csv` được nộp.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình theo nhóm câu hỏi — dữ liệu thật từ `outputs/graphrag_vs_flatrag_summary.csv`):

| Tiêu chí đánh giá | Nhóm          | Flat RAG   | GraphRAG   | Δ             | Nhận xét                                                |
| ---------------------- | -------------- | ---------- | ---------- | -------------- | --------------------------------------------------------- |
| Comprehensiveness      | factoid        | 5.0        | 5.0        | 0              | Ngang nhau                                                |
| Comprehensiveness      | cross-doc      | 3.0        | 5.0        | +2.0           | GraphRAG tốt hơn rõ rệt                               |
| Comprehensiveness      | multi-hop      | 1.0        | 2.0        | +1.0           | Cả 2 đều kém — xem phân tích Judge bên dưới     |
| Faithfulness           | factoid        | 5.0        | 5.0        | 0              | —                                                        |
| Faithfulness           | cross-doc      | 4.0        | 5.0        | +1.0           | GraphRAG tốt hơn                                        |
| Faithfulness           | multi-hop      | 1.0        | 5.0        | **+4.0** | Chênh lệch lớn nhất trong toàn bộ benchmark         |
| Multi-hop reasoning    | factoid        | 5.0        | 5.0        | 0              | —                                                        |
| Multi-hop reasoning    | cross-doc      | 3.0        | 5.0        | +2.0           | GraphRAG tốt hơn                                        |
| Multi-hop reasoning    | multi-hop      | 1.0        | 2.0        | +1.0           | Cả 2 đều kém                                          |
| Latency (s)            | factoid        | 5.49       | 5.23       | −0.3          | Ngang nhau, GraphRAG thậm chí nhỉnh hơn ở nhóm này |
| Latency (s)            | cross-doc      | 5.55       | 22.13      | +16.6          | GraphRAG chậm hơn ~4x                                   |
| Latency (s)            | multi-hop      | 11.78      | 20.92      | +9.1           | GraphRAG chậm hơn ~1.8x                                 |
| Token usage            | tất cả nhóm | 1540–2693 | 1642–2742 | +~100          | GraphRAG tốn thêm ít token                             |

**Nhận xét tổng quan:** GraphRAG thắng rõ và nhất quán ở `cross-doc` (đúng lý thuyết — cần tổng hợp nhiều nguồn). Ở nhóm `multi-hop`, cả hai phương pháp đều bị Judge chấm rất thấp (1–2/5) — xem phân tích Ca lỗi 2 bên dưới, đây hoá ra là dấu hiệu của **Judge không ổn định** hơn là lỗi retrieval thật. Latency GraphRAG cao hơn Flat RAG 0–4x tuỳ câu hỏi, phản ánh đúng bản chất hybrid (chi phí cộng dồn: vector retrieval + seed-extraction LLM call + graph traversal).

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG yếu hơn rõ rệt (GraphRAG thành công) — G03 (cross-doc):**

   - *Câu hỏi:* "...describe how First Orion Hiya (PARTNERED_WITH Neustar TNS) and Fidelity National Information Services (ACQUIRED Worldpay) relate to the broader tech industry trend they represent." — Judge: Comprehensiveness 3→5, Faithfulness 4→5, Multi-hop 3→5.
   - *Tại sao Flat RAG yếu hơn:* Cả hai retrieve đúng chunk chứa xu hướng ngành ("ongoing digitization... diversify technology services"), nhưng câu trả lời Flat RAG **hedging quá mức** — nhiều lần tự kiểm tra rồi kết luận "the supplied text does not explicitly link these specific corporate actions to that broader industry trend", tuân thủ đúng chỉ dẫn "answer only from context" nhưng bị Judge chấm thiếu tổng hợp/thiếu thuyết phục.
   - *GraphRAG giải quyết thế nào:* Context `=== GRAPH ===` liệt kê rõ 2 cạnh `PARTNERED_WITH`/`ACQUIRED` dưới dạng triple có cấu trúc, giúp model tự tin hơn khi suy luận liên kết 2 sự kiện với xu hướng ngành nêu trong đoạn `VECTOR`, đưa ra câu trả lời tổng hợp mạch lạc hơn — dù **cùng một tập chunk nguồn** với Flat RAG. Kết quả này lặp lại nhất quán ở cả 2 lần chạy pipeline độc lập trong buổi (lần 1: 4→5, lần cuối: 3→5), nên đây là phát hiện đáng tin cậy chứ không phải nhiễu ngẫu nhiên.
2. **Ca lỗi ở tầng Đánh giá (LLM-as-a-Judge) — G04 (multi-hop), phát hiện quan trọng hơn dự kiến:**

   - *Câu hỏi:* "...describe how Freshworks Inc. (DEVELOPED AI-powered Customer Service Suite) and Eseye (DEVELOPED integrated cellular IoT connectivity solutions) relate to the broader tech industry trend they represent." — Judge: Flat = 1/1/1, Graph = 2/5/2.
   - *Phát hiện qua đọc trực tiếp `flat_judge_rationale`:* Rationale viết: *"The candidate provides a detailed analysis... accurately cites specific chunks... The reasoning is clear and follows the constraints..."* — mô tả này rõ ràng là **đánh giá tích cực**, tương xứng với điểm 4–5, nhưng trường JSON `comprehensiveness`/`multi_hop_reasoning` mà Judge trả về lại là **1**. Đây là **mâu thuẫn giữa rationale (text) và score (number)** trong cùng 1 lần gọi Judge — một failure mode ở tầng đánh giá, không phải ở tầng retrieval của Flat RAG hay GraphRAG.
   - *Nguyên nhân:* `JUDGE_SYSTEM`/`judge_answer()` yêu cầu model trả JSON strict gồm cả rationale lẫn số điểm trong 1 lần sinh — với model reasoning dài dòng (`<think>` trace), điểm số cuối có thể bị "lệch nhịp" so với phần rationale do model tự mâu thuẫn trong quá trình suy luận nhiều bước, hoặc do neo điểm số vào so sánh ngầm với câu trả lời kia (anchoring) thay vì đánh giá độc lập.
   - *Đề xuất khắc phục:* (1) Tách rationale và score thành 2 lần gọi riêng (score trước, rationale sau) để giảm anchoring; (2) Thêm bước self-consistency: gọi Judge 2–3 lần lấy trung vị điểm số; (3) Validate: nếu rationale chứa các cụm tích cực rõ ràng ("accurately", "clear", "detailed") mà score ≤ 2, tự động flag để review thủ công thay vì tin tuyệt đối vào con số.
3. **Ca lỗi GraphRAG thất bại — G04 (multi-hop):**

   - *Câu hỏi:* "...describe how Freshworks Inc. (DEVELOPED AI-powered Customer Service Suite) and Eseye (DEVELOPED integrated cellular IoT connectivity solutions) relate to the broader tech industry trend they represent." — Judge: Comprehensiveness 5→2, Multi-hop 5→2.
   - *Nguyên nhân thật (đã đọc full transcript 2 câu trả lời):* Đây **không phải** do super-node cap hay thiếu edge (đồ thị quá thưa, degree=1, không có supernode_event nào bị kích hoạt). Nguyên nhân là **hiệu ứng ngược của định dạng context có cấu trúc**: GraphRAG nhận thêm khối `=== GRAPH ===` liệt kê 2 triple tách biệt (`Freshworks -> AI-powered Customer Service Suite`, `Eseye -> IoT solutions`) rõ ràng KHÔNG có cạnh nào nối 2 triple này với nhau hay với node xu hướng ngành — điều này khiến model "nhìn thấy" rõ sự thiếu liên kết trong đồ thị và **chủ động từ chối suy luận** ("the provided text does not explicitly link... evidence is insufficient"). Ngược lại Flat RAG chỉ nhận text thuần, không có tín hiệu cấu trúc nào nhắc nhở về việc "thiếu liên kết", nên tự do tổng hợp một kết luận mạch lạc hơn (dù thực chất cũng chỉ suy luận từ cùng 1 chunk xu hướng ngành).
   - *Đề xuất khắc phục:* (1) Điều chỉnh `ANSWER_SYSTEM` prompt để khuyến khích tổng hợp liên kết ngầm ("infer reasonable connections between separate graph facts and background trend context") thay vì chỉ yêu cầu "answer only from context" một cách cứng nhắc; (2) Bổ sung thêm 1-hop mở rộng quanh mỗi seed entity (hiện `max_hops=2` nhưng đồ thị thưa không có gì để mở rộng) khi triển khai ở scale lớn hơn để GraphRAG thực sự có nhiều cạnh liên kết hơn Flat RAG, thay vì chỉ nhỉnh hơn ở phần trình bày.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

> **Trade-offs, Agent Control & Scale 350MB:**
>
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*

- **Đánh đổi Quality vs Cost vs Latency (số liệu thật, lần chạy cuối):** GraphRAG chậm hơn Flat RAG 0–16.6 giây/câu tuỳ nhóm (ngang nhau ở `factoid`, chậm hơn ~4x ở `cross-doc`) và tốn thêm ~100 token/câu, vì kiến trúc hybrid **cộng dồn** chi phí (vẫn giữ nguyên vector retrieval + thêm LLM call trích seed entity + BFS traversal Cypher) chứ không thay thế. Đổi lại, chất lượng thắng rõ và nhất quán ở nhóm `cross-doc` (+2 điểm/tiêu chí, lặp lại ở cả 2 lần chạy độc lập) — đây là nơi GraphRAG thực sự đáng giá. Với graph thưa (12 edge, mọi node degree=1), **Indexing Overhead** của GraphRAG (bulk insert Neo4j + entity resolution + xây constraint/index) chỉ được đền bù xứng đáng ở nhóm câu hỏi cross-doc; bài học quan trọng: GraphRAG chỉ đáng chi phí khi câu hỏi thực sự đòi hỏi tổng hợp nhiều nguồn, không phải mọi loại câu hỏi.
- **Quyết định từ chối/điều chỉnh đề xuất của AI Coding Agent (Claude) trong buổi làm việc này:** Khi pipeline gặp lỗi rate-limit lần đầu, Agent đề xuất chạy lại toàn bộ notebook từ đầu (bao gồm re-run Coreference 15-20 phút mỗi lần) — cách tiếp cận "brute-force retry" này lặp lại 2 lần trước khi được thay bằng cơ chế **checkpoint pickle** (lưu kết quả `coref_df`/`raw_triples_df` ra đĩa, resume thay vì tính lại) — quyết định đúng vì đã tránh tốn thêm hàng chục phút và hàng chục nghìn token Groq ở các lần chạy tiếp theo. Đây là minh chứng cho việc **không nên chấp nhận mù quáng hướng tiếp cận đầu tiên của AI Agent** khi chi phí thử-sai (token/thời gian) là hữu hạn và có thể đo được.
- **Giải pháp khi scale lên 350MB (~100.000 bài báo):** Bottleneck đầu tiên **không phải OOM/RAM** như giả định phổ biến, mà là **rate limit của LLM provider** — thực nghiệm hôm nay cho thấy chỉ với 400 chunk (Coreference + Extraction), pipeline đã 3 lần chạm giới hạn TPD (200.000 token/ngày) trên 2 model Groq khác nhau (`gpt-oss-120b`, `gpt-oss-20b`) trong cùng 1 ngày. Ở scale 100.000 bài báo (~250x), chi phí LLM call sẽ vượt xa mọi free-tier hiện có. Giải pháp: (1) **Batch theo hàng đợi đa ngày/đa API-key** thay vì 1 lần chạy liên tục; (2) **worker queue** với backoff động theo header rate-limit trả về (thay vì retry cố định); (3) **downgrade model** cho các bước ít quan trọng (coreference dùng model nhỏ, chỉ extraction quan trọng dùng model lớn) để rải tải qua nhiều quota bucket riêng biệt; (4) **Entity Resolution** chuyển từ `IndexFlatIP` (so khớp toàn bộ) sang **HNSW/IVF index** để giữ thời gian truy vấn gần-tuyến-tính khi số entity tăng lên hàng trăm nghìn.

---

## PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng            | Module tương ứng | Hàm / Khối code cụ thể                       | Quan sát thực tế & Đánh giá                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------- | ------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Conservative Coreference**       | Module 1            | `resolve_coref_batch()`, `run_coref()`       | Trên 400 chunk: 182 giữ nguyên do ambiguous (`unresolved_mentions`), 123 chunk bị thay đổi. Rule "chỉ resolve khi antecedent rõ trong cùng chunk" hoạt động đúng ở mức object-level, nhưng cách LLM thay thế nguyên văn cụm dài (kèm viết tắt trong ngoặc) có thể gây lệch chuẩn hoá tên ở bước Entity Resolution sau này (xem Câu 1).                           |
| **Schema & Allowlist Guard**       | Module 2            | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`    | Hoạt động đúng: extraction chỉ giữ lại quan hệ nằm trong 8 loại cho phép (`if rel not in ALLOWED_RELATIONS: continue`), giúp lọc bỏ các quan hệ "sáng tạo" mà model có thể tự bịa ra ngoài schema.                                                                                                                                                                             |
| **Bulk Cypher Ingestion**          | Module 2            | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND $rows AS row` chạy đúng batch 1000, `graph_checks()` xác nhận 0 cạnh thiếu `source_chunk_id`/`published_date` trên toàn bộ 12 cạnh thật.                                                                                                                                                                                                                                      |
| **Entity Resolution & Union-Find** | Module 3            | `build_resolution_map()`, `UF`               | Với dataset mẫu (24 entity phân biệt tích luỹ trong Neo4j), similarity cao nhất đo được chỉ 0.3987 — đã thử hạ threshold từ 0.90 xuống 0.30 để tìm cặp biên nhưng vẫn không có cặp nào lọt qua trong batch triples cuối cùng, xác nhận đây là đặc điểm dữ liệu (ít trùng lặp tên) chứ không phải lỗi code (xem Câu 2).                               |
| **Super-node Degree Cap**          | Module 4            | `retrieve_graph_context()`                     | Chưa quan sát được kích hoạt thật (mọi node degree=1, dưới ngưỡng`SUPER_NODE_DEGREE=100` rất xa) — logic đã review code là đúng nhưng cần dataset lớn hơn để kiểm chứng bằng thực nghiệm.                                                                                                                                                                                |
| **LLM-as-a-Judge Evaluation**      | Module 5            | `judge_answer()`                               | Chạy đúng trên 5 câu Golden tự sinh từ dữ liệu graph thật, phát hiện GraphRAG thắng rõ và nhất quán ở`cross-doc`; đồng thời phát hiện chính bản thân Judge có thể **mâu thuẫn giữa rationale và score** trong 1 lần gọi (case G04, xem Câu 4) — một rủi ro chỉ lộ ra khi đọc kỹ output thật, không thể thấy nếu chỉ nhìn số điểm tổng hợp. |

---

### 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Chuỗi lỗi liên hoàn khi chạy Module 2 (Triple Extraction) trên máy local thay vì Colab:
  1. Dataset thật (`HackerNoon/tech-company-news-data-dump`) có schema khác giả định trong code mẫu — chỉ có cột `description`, không có `text/content/article/body/story` → `pick_col()` raise `KeyError`.
  2. `GROQ_MODEL=llama-3.3-70b-versatile` không còn tồn tại trên API Groq hiện tại (model đã bị deprecate) → lỗi `404 model_not_found`.
  3. Sau khi đổi sang `openai/gpt-oss-120b`, chạy được nhưng **hết quota 200.000 token/ngày (TPD)** giữa chừng batch 12/100 vì cell Coreference (80 batch) và Extraction (100 batch) dùng chung 1 model trong cùng ngày → 93/100 batch extraction fail `429 rate_limit_exceeded`, đồ thị chỉ còn 8 cạnh mồ côi.
  4. Đổi sang `openai/gpt-oss-20b` (quota riêng) để né rate-limit, nhưng model nhỏ hơn trả JSON kém ổn định hơn — trường `items`/`relations` đôi khi chứa `string` thay vì `object`, gây `AttributeError: 'str' object has no attribute 'get'` làm crash toàn bộ `run_extraction()`.
- **Cách xử lý:** (1) mở rộng danh sách cột ứng viên trong `pick_col()` thêm `"description"`; (2) liệt kê model thật khả dụng qua `client.models.list()` thay vì tin vào giá trị mặc định trong `.env.example`; (3) tách extraction sang model có quota riêng biệt với coreference thay vì dùng chung 1 model cho mọi tác vụ trong ngày; (4) thêm `isinstance(item, dict)` / `isinstance(x, dict)` guard trước khi gọi `.get()` để bỏ qua phần tử JSON dị dạng thay vì crash cả batch.
- **Bài học:** Lab guide viết cho Colab + dataset "giả định" không tự động chạy đúng trên môi trường/dataset thật khác — luôn cần validate schema thật và giới hạn hạ tầng (rate limit theo model, không phải chỉ theo tài khoản) trước khi tin vào cấu hình mặc định.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

> *Phần này cần điền theo đồ án cá nhân thật của bạn — nội dung dưới đây là khung gợi ý dựa trên bài học rút ra từ lab, bạn nên thay bằng bài toán cụ thể bạn đang làm.*

- **Tên đồ án / Dự án:** [Điền tên đồ án của bạn]
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài học lớn nhất từ lab này là **GraphRAG chỉ đáng chi phí (latency gấp 2–3x, thêm bước Entity Resolution + Ingestion) khi đồ thị đủ dày và câu hỏi thực sự cần nối nhiều quan hệ rời rạc** (nhóm `cross-doc` thắng rõ trong thực nghiệm). Nếu đồ án của bạn chủ yếu là câu hỏi tra cứu 1 sự kiện đơn (factoid), Flat/Hybrid RAG đã đủ và rẻ hơn nhiều — chỉ nên đầu tư GraphRAG nếu miền dữ liệu có nhiều thực thể lặp lại, quan hệ rõ ràng (vd: hồ sơ pháp lý liên công ty, quan hệ nhân sự nội bộ, chuỗi cung ứng).
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: xác định 3–5 loại thực thể cốt lõi của miền dữ liệu (tương tự `Company/Person/Technology` ở lab), tránh quá nhiều loại gây loãng schema.
  - Relations: giới hạn allowlist rõ ràng (như `ALLOWED_RELATIONS` ở lab) thay vì để LLM tự do đặt tên quan hệ — bài học từ Module 2 cho thấy allowlist giúp lọc nhiễu hiệu quả.
- **Chiến lược xử lý Super-node & Entity Resolution:** Áp dụng nguyên bản `SUPER_NODE_DEGREE`/`SUPER_NODE_EDGE_CAP` nhưng **hiệu chỉnh ngưỡng theo phân bố degree thật của dữ liệu đồ án** (lab dùng 100 vì dataset mẫu quá thưa để kiểm chứng — cần benchmark lại trên dữ liệu thật của đồ án trước khi tin ngưỡng mặc định). Với Entity Resolution, giữ nguyên kiến trúc 4 tầng (manual alias → vector ANN → lexical guard → Union-Find) nhưng cân nhắc hạ threshold nếu domain có nhiều biến thể tên hơn dataset tin tức công nghệ.

---

## TỰ ĐÁNH GIÁ

| Tiêu chí                                   | Điểm tự chấm (1–5) | Ghi chú                                                                                                                                                                                                                                                                 |
| -------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Mức độ hiểu bài giảng GraphRAG         | 4                       | Giải thích được vì sao GraphRAG thắng ở`cross-doc` nhưng thua ở `multi-hop` bằng cách đọc trực tiếp reasoning trace của model, không chỉ dựa vào số điểm Judge.                                                                              |
| Khả năng kiểm soát AI Coding Agent       | 4                       | Không chấp nhận mù quáng cách tiếp cận đầu tiên (brute-force re-run); yêu cầu Agent chẩn đoán tận gốc từng lỗi (schema, model 404, rate-limit, JSON malformed) thay vì che giấu bằng try/except tràn lan.                                        |
| Chất lượng đồ thị tri thức xây dựng | 3                       | Graph đúng schema, đủ provenance, 0 lỗi — nhưng khách quan là thưa (12 edge/24 node, degree=1 toàn bộ) do giới hạn scale-guard + tính chất dữ liệu mẫu, chưa đủ để kiểm chứng Super-node Mitigation bằng thực nghiệm thật.                 |
| Khả năng phân tích và debug hệ thống  | 5                       | Xử lý được chuỗi lỗi phức tạp: schema dataset sai giả định, model deprecated, rate-limit TPD, JSON không ổn định, Neo4j connection timeout, và một bug tự gây ra (ghi đè cell 0) — đều truy ra nguyên nhân gốc thay vì workaround bề mặt. |
