# ============================================================

st.markdown("### Ask your notes")
st.caption("Answers are generated from retrieved passages. Check the source panel when accuracy matters.")

chat_header_columns = st.columns([4, 1.3])
with chat_header_columns[0]:
    st.caption(
        f"Questions used this session: {st.session_state.question_count}/"
        f"{MAX_QUESTIONS_PER_SESSION}"
    )
with chat_header_columns[1]:
    if st.button("Clear chat", use_container_width=True):
        clear_conversation()
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            display_sources(message["sources"])

question = None
if st.session_state.question_count < MAX_QUESTIONS_PER_SESSION:
    question = st.chat_input(
        "Ask a question about your uploaded PDFs…",
        max_chars=MAX_QUESTION_CHARACTERS,
    )
else:
    st.warning(
        "You've reached the question limit for this session. "
        "This basic limit is session-based and can reset when a new session starts."
    )


# ============================================================
# 17. HANDLE QUESTIONS
# ============================================================

if question:
    question = question.strip()
    if not question:
        st.stop()

    if len(question) > MAX_QUESTION_CHARACTERS:
        st.warning(f"Questions must be no longer than {MAX_QUESTION_CHARACTERS} characters.")
        st.stop()

    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("You've reached the question limit for this session.")
        st.stop()

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-6:]
        if message.get("role") in ("user", "assistant")
    ]

    st.session_state.question_count += 1
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            client = get_api_client(GROQ_API_KEY)
            model = load_embedding_model()

            with st.spinner("Understanding your question…"):
                search_query = rewrite_search_query(question, history, client)

            with st.spinner("Searching your documents…"):
                results = retrieve_chunks(
                    search_query,
                    chunks,
                    embeddings,
                    model,
                    top_k=TOP_K,
                )

            if not results:
                answer = (
                    "I couldn't find relevant passages in your uploaded documents "
                    "to answer that question."
                )
                source_records = []
            else:
                with st.spinner("Preparing your answer…"):
                    answer = generate_answer(question, results, client, history)

                source_records = [
                    {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "score": score,
                        "text": chunk["text"],
                    }
                    for _, score, chunk in results
                ]

            st.markdown(answer)
            if source_records:
                display_sources(source_records)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_records,
            })

        except Exception as error:
            st.error(
                f"The request failed ({type(error).__name__}). "
                "Please try again or check the app logs."
            )
            print(repr(error))
