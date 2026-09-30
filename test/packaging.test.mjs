import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const manifest = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const validator = join(root, "scripts", "validate-release.mjs");

await test("the public package declares gallery eligibility and host-provided peers", () => {
  assert.equal(manifest.name, "pi-cocore");
  assert.equal(manifest.type, "module");
  assert.equal(manifest.license, "GPL-3.0-or-later");
  assert.ok(manifest.keywords.includes("pi-package"));
  assert.equal(manifest.repository.url, "git+https://github.com/randoneering/pi-cocore.git");
  for (const dependency of ["@earendil-works/pi-ai", "@earendil-works/pi-coding-agent"]) {
    assert.equal(manifest.peerDependencies?.[dependency], "*", dependency);
    assert.equal(manifest.dependencies?.[dependency], undefined, "do not bundle host packages");
  }
});

await test("npm publishes only the extension, metadata, and license notices", () => {
  const result = spawnSync("npm", ["pack", "--dry-run", "--json", "--ignore-scripts"], {
    cwd: root, encoding: "utf8", timeout: 30_000,
  });
  assert.equal(result.status, 0, result.stderr);
  const [packed] = JSON.parse(result.stdout);
  assert.deepEqual(packed.files.map((file) => file.path).sort(), [
    "LICENSE", "NOTICE", "README.md", "extensions/cocore.ts", "package.json",
  ]);
  assert.deepEqual(packed.bundled, []);
  assert.deepEqual(manifest.pi.extensions, ["./extensions/cocore.ts"]);
  for (const entry of manifest.pi.extensions) {
    assert.ok(packed.files.some((file) => file.path === entry.replace(/^\.\//, "")));
  }
});

function validateRelease(tag, version) {
  const cwd = mkdtempSync(join(tmpdir(), "pi-cocore-release-test-"));
  writeFileSync(join(cwd, "package.json"), JSON.stringify({ name: "pi-cocore", version }));
  const output = join(cwd, "github-output");
  const result = spawnSync(process.execPath, [validator], {
    cwd, encoding: "utf8", timeout: 10_000,
    env: { ...process.env, RELEASE_TAG: tag, GITHUB_OUTPUT: output },
  });
  return {
    result,
    output,
    manifestAfter: readFileSync(join(cwd, "package.json"), "utf8"),
  };
}

await test("a matching stable release publishes to latest without rewriting its version", () => {
  const { result, output, manifestAfter } = validateRelease("v1.0.0", "1.0.0");
  assert.equal(result.status, 0, result.stderr);
  assert.match(readFileSync(output, "utf8"), /^version=1\.0\.0\ndist-tag=latest\n$/);
  assert.equal(manifestAfter, '{"name":"pi-cocore","version":"1.0.0"}');
});

await test("a matching prerelease publishes to next", () => {
  const { result, output } = validateRelease("v1.1.0-rc.2", "1.1.0-rc.2");
  assert.equal(result.status, 0, result.stderr);
  assert.match(readFileSync(output, "utf8"), /^version=1\.1\.0-rc\.2\ndist-tag=next\n$/);
});

await test("a tag/version mismatch blocks publication", () => {
  const { result, output } = validateRelease("v1.0.1", "1.0.0");
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /does not match package\.json/);
  assert.equal(existsSync(output), false);
});

await test("malformed tags cannot supply publish arguments", () => {
  for (const tag of ["1.0.0", "v01.0.0", "v1.0.0-01", "v1.0.0;echo unsafe"]) {
    const { result, output } = validateRelease(tag, "1.0.0");
    assert.notEqual(result.status, 0, tag);
    assert.match(result.stderr, /Invalid release tag/, tag);
    assert.equal(existsSync(output), false, tag);
  }
});
