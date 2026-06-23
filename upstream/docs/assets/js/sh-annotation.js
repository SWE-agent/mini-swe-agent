// Source: https://github.com/srl-labs/learn-srlinux/pull/133/files
// this script is used to remove extra leading space when annotating shell code blocks ending with `\`
// character. See https://github.com/squidfunk/mkdocs-material/issues/3846 for more info.
document$.subscribe(() => {
  const tags = document.querySelectorAll("code .se");
  tags.forEach((tag) => {
    if (tag.innerText.startsWith("\\")) {
      tag.innerText = "\\";
    }
  });
});
