@app.post(
    "/smartroom",
    response_model=SmartRoomOutput,
    status_code=status.HTTP_200_OK,
)
async def smartroom_endpoint(payload: SmartRoomInput) -> SmartRoomOutput:
    """
    Main Smart‑Room endpoint:
    1. Parse the command → lights_on/off + optional colour.
    2. Ask Groq’s LLM to explain the reasoning.
    3. Return a structured SmartRoomOutput.
    """

    # 1️⃣ Parse the command
    parsed: ActionResult = parse_command(payload.command)
    action = parsed.combined
    log.info("Parsed command → %s", action)

    # 2️⃣ Build the LLM prompt
    messages = [
        {
            "role": "user",
            "content": (
                f"Motion: {payload.motion}\n"
                f"Noise: {payload.noise}\n"
                f"Command: {payload.command}\n"
                f"Action chosen: {action}\n\n"
                "Explain the reasoning behind this action as the Smart‑Room AI."
            ),
        }
    ]

    # 3️⃣ Call Groq (with retries)
    try:
        ai_text = await _call_groq_llm(messages)
    except HTTPException as exc:
        # Bubble up the HTTPException so FastAPI returns a proper 4xx/5xx
        raise exc
    except Exception as exc:
        log.exception("Unexpected Groq error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected Groq error: {exc}",
        )

    # 4️⃣ Return structured output
    return SmartRoomOutput(
        status="ok",
        action=action,
        ai_response=ai_text,
    )
