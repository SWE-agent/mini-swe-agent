document.addEventListener("DOMContentLoaded", function () {
  const gifContainers = document.querySelectorAll(".gif-container");

  gifContainers.forEach((container) => {
    const img = container.querySelector("img[data-gif]");

    if (img) {
      const pngSrc = img.src;
      const gifSrc = img.getAttribute("data-gif");
      let isGif = false;

      // Create play button dynamically
      const playButton = document.createElement("button");
      playButton.className = "gif-play-button";
      playButton.setAttribute("aria-label", "Play demo");
      container.appendChild(playButton);

      function toggleGif() {
        if (!isGif) {
          // Switch to GIF
          img.src = gifSrc;
          isGif = true;
          container.classList.add("playing");
        } else {
          // Switch back to PNG
          img.src = pngSrc;
          isGif = false;
          container.classList.remove("playing");
        }
      }

      // Play button click
      playButton.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        toggleGif();
      });

      // Image click (play or pause)
      img.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        toggleGif();
      });
    }
  });
});
