import { appendFileSync, readFileSync } from "node:fs";

const tagPattern = /^v((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?)$/;

try {
  const tag = process.env.RELEASE_TAG;
  const match = typeof tag === "string" ? tagPattern.exec(tag) : null;
  const invalidNumericPrerelease = match?.[2]?.split(".").some(
    (identifier) => /^\d+$/.test(identifier) && /^0\d/.test(identifier),
  );
  if (!match || invalidNumericPrerelease) {
    throw new Error("Invalid release tag. Use v<major>.<minor>.<patch> with an optional prerelease suffix.");
  }

  const manifest = JSON.parse(readFileSync("package.json", "utf8"));
  const version = match[1];
  if (version !== manifest.version) {
    throw new Error(`Release tag ${tag} does not match package.json version ${manifest.version}. Commit the version change before releasing.`);
  }

  const distTag = match[2] ? "next" : "latest";
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(process.env.GITHUB_OUTPUT, `version=${version}\ndist-tag=${distTag}\n`);
  }
  console.log(`Release ${manifest.name}@${version} uses npm dist-tag ${distTag}.`);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
