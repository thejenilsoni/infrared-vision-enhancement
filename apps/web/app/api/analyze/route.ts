import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const inferenceUrl =
  process.env.INFERENCE_API_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const form = await request.formData();
    const response = await fetch(`${inferenceUrl}/v1/analyze`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(45_000)
    });

    const contentType =
      response.headers.get("content-type") ?? "application/json";
    const body = await response.arrayBuffer();

    return new NextResponse(body, {
      status: response.status,
      headers: { "content-type": contentType }
    });
  } catch {
    return NextResponse.json(
      {
        detail:
          "Inference service unavailable. Start the Python service or use docker compose."
      },
      { status: 503 }
    );
  }
}
