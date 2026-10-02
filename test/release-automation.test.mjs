import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

function readConfig(relativePath) {
  const path = join(root, relativePath);
  return existsSync(path) ? parse(readFileSync(path, "utf8")) : {};
}

const workflow = readConfig(".github/workflows/release-drafter.yml");
const config = readConfig(".github/release-drafter.yml");

// Exercise the configured expressions against PR fixtures, without a GitHub token.
function labelsForPr({ branch = "", title = "", body = "" }) {
  return (config.autolabeler ?? [])
    .filter((rule) => ["branch", "title", "body"].some((field) => {
      const value = { branch, title, body }[field];
      return (rule[field] ?? []).some((pattern) => {
        const end = pattern.lastIndexOf("/");
        const expression = new RegExp(pattern.slice(1, end), pattern.slice(end + 1));
        return expression.test(value);
      });
    }))
    .map((rule) => rule.label);
}

function versionBump(labels) {
  for (const level of ["major", "minor", "patch"]) {
    if (config["version-resolver"]?.[level]?.labels.some((label) => labels.includes(label))) {
      return level;
    }
  }
  return config["version-resolver"]?.default;
}

await test("release drafting runs after main changes and allows manual refresh", () => {
  assert.deepEqual(workflow.on?.push?.branches, ["main"]);
  assert.ok(Object.hasOwn(workflow.on ?? {}, "workflow_dispatch"));
});

await test("PR label updates handle title edits without running PR code with write access", () => {
  const event = workflow.on?.pull_request_target;
  assert.deepEqual(event?.branches, ["main"]);
  for (const type of ["opened", "reopened", "synchronize", "edited"]) {
    assert.ok(event.types.includes(type), `missing PR event: ${type}`);
  }
  const jobs = Object.values(workflow.jobs ?? {});
  assert.equal(jobs.length, 1);
  const [job] = jobs;
  assert.equal(workflow.permissions?.contents, "read");
  assert.equal(job.permissions?.contents, "write");
  assert.equal(job.permissions?.["pull-requests"], "write");
  assert.equal(job.if, "github.repository == 'randoneering/pi-cocore'");
  assert.equal(job.steps.length, 1, "privileged job must only run the drafting action");
  assert.match(job.steps[0].uses, /^release-drafter\/release-drafter@[a-f0-9]{40}$/);
  assert.equal(job.steps[0].run, undefined, "do not execute PR-controlled shell commands");
  assert.equal(job.steps[0].env.GITHUB_TOKEN, "${{ secrets.GITHUB_TOKEN }}");
});

await test("PR fixtures select labels, release categories, and version bumps", () => {
  const cases = [
    [{ branch: "feat/model-picker" }, "feat", "Features", "minor"],
    [{ title: "fix: recover tool calls" }, "fix", "Bug Fixes", "patch"],
    [{ branch: "docs/tool-support" }, "docs", "Documentation", "patch"],
    [{ title: "refactor(stream): split parser" }, "refactor", "Refactoring", "patch"],
    [{ branch: "test/agent-loop" }, "test", "Testing", "patch"],
    [{ title: "chore: update tooling" }, "chore", "Chores", "patch"],
    [{ branch: "perf/stream" }, "perf", "Performance", "patch"],
    [{ title: "style: format extension" }, "style", "Style", "patch"],
    [{ branch: "ci/release-drafter" }, "ci", "CI/CD", "patch"],
    [{ title: "build: update dependencies" }, "build", "Build", "patch"],
    [{ title: "revert: undo parser change" }, "revert", "Reverts", "patch"],
  ];
  for (const [pr, label, category, bump] of cases) {
    const labels = labelsForPr(pr);
    assert.ok(labels.includes(label), JSON.stringify(pr));
    assert.equal(config.categories?.find((item) => item.labels.includes(label))?.title, category);
    assert.equal(versionBump(labels), bump);
  }
});

await test("breaking changes override feature and fix version bumps", () => {
  for (const pr of [
    { title: "feat(provider)!: change tool contract" },
    { title: "fix: correct routing", body: "Details\nBREAKING CHANGE: rename a command" },
  ]) {
    const labels = labelsForPr(pr);
    assert.ok(labels.includes("breaking"));
    assert.equal(versionBump(labels), "major");
  }
  assert.equal(versionBump(["minor", "fix"]), "minor");
  assert.equal(versionBump(["major", "minor", "fix"]), "major");
  assert.equal(versionBump([]), "patch", "unlabelled changes still receive a release version");
});

await test("draft release names, tags, and notes use the resolved version and merged changes", () => {
  const values = {
    RESOLVED_VERSION: "1.2.3", CHANGES: "- Fix tool calls (#5) by @contributor",
    TITLE: "Fix tool calls", NUMBER: "5", AUTHOR: "contributor",
  };
  const render = (template = "") => template.replace(/\$(RESOLVED_VERSION|CHANGES|TITLE|NUMBER|AUTHOR)\b/g,
    (_match, name) => values[name]);
  assert.equal(render(config["name-template"]), "v1.2.3");
  assert.equal(render(config["tag-template"]), "v1.2.3");
  assert.match(render(config.template), /Fix tool calls \(#5\) by @contributor/);
  assert.equal(render(config["change-template"]), "- Fix tool calls (#5) by @contributor");
});
