// ABOUTME: Vercel serverless function for reading/writing sentence labels.
// ABOUTME: Stores per-user label files in Cloudflare R2 (S3-compatible).

import { S3Client, GetObjectCommand, PutObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";

const s3 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

const BUCKET = process.env.R2_BUCKET_NAME || "freedomlabels";

async function getObject(key) {
  try {
    const resp = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: key }));
    return JSON.parse(await resp.Body.transformToString());
  } catch (e) {
    if (e.name === "NoSuchKey" || e.$metadata?.httpStatusCode === 404) return null;
    throw e;
  }
}

async function putObject(key, data) {
  await s3.send(new PutObjectCommand({
    Bucket: BUCKET,
    Key: key,
    Body: JSON.stringify(data),
    ContentType: "application/json",
  }));
}

export default async function handler(req, res) {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, PUT, PATCH, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const { user, action } = req.query;

    // GET /api/labels?user=sebastian — fetch one user's labels
    if (req.method === "GET" && user && action !== "all") {
      const data = await getObject(`labels/${user}.json`);
      return res.json(data || {});
    }

    // GET /api/labels?action=all — fetch all users' labels merged
    if (req.method === "GET" && action === "all") {
      const list = await s3.send(new ListObjectsV2Command({
        Bucket: BUCKET, Prefix: "labels/",
      }));
      const allLabels = {};
      for (const obj of list.Contents || []) {
        const username = obj.Key.replace("labels/", "").replace(".json", "");
        const data = await getObject(obj.Key);
        if (data) {
          for (const [id, label] of Object.entries(data)) {
            if (!allLabels[id]) allLabels[id] = {};
            allLabels[id][username] = label;
          }
        }
      }
      return res.json(allLabels);
    }

    // GET /api/labels?action=users — list all users
    if (req.method === "GET" && action === "users") {
      const list = await s3.send(new ListObjectsV2Command({
        Bucket: BUCKET, Prefix: "labels/",
      }));
      const users = (list.Contents || []).map(o =>
        o.Key.replace("labels/", "").replace(".json", "")
      );
      return res.json({ users });
    }

    // PUT /api/labels — save labels for a user (bulk)
    if (req.method === "PUT") {
      const body = req.body;
      if (!body?.user || !body?.labels) {
        return res.status(400).json({ error: "Missing user or labels" });
      }
      const existing = await getObject(`labels/${body.user}.json`) || {};
      const merged = { ...existing, ...body.labels };
      await putObject(`labels/${body.user}.json`, merged);
      return res.json({ ok: true, count: Object.keys(merged).length });
    }

    // PATCH /api/labels — save a single label (real-time)
    // Body: { user: "email", id: "sentence-id", label: { label: "positive_liberty", ... } }
    if (req.method === "PATCH") {
      const body = req.body;
      if (!body?.user || !body?.id || !body?.label) {
        return res.status(400).json({ error: "Missing user, id, or label" });
      }
      const existing = await getObject(`labels/${body.user}.json`) || {};
      existing[body.id] = body.label;
      await putObject(`labels/${body.user}.json`, existing);
      return res.json({ ok: true, id: body.id });
    }

    return res.status(400).json({ error: "Invalid request" });

  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: e.message });
  }
}
