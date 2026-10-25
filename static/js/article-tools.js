"use strict";

function revealArticleAnchor() {
    let id;
    try {
        id = decodeURIComponent(window.location.hash.slice(1));
    } catch {
        return;
    }
    const target = document.getElementById(id);
    if (!target) return;

    for (let parent = target; parent; parent = parent.parentElement) {
        if (parent.tagName === "DETAILS") parent.open = true;
    }
    requestAnimationFrame(() => target.scrollIntoView());
}

window.addEventListener("hashchange", revealArticleAnchor);
revealArticleAnchor();

document.addEventListener("click", (event) => {
    const link = event.target.closest("a");
    if (!link || event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

    if (link.dataset.videoId) {
        if (!/^[A-Za-z0-9_-]{11}$/.test(link.dataset.videoId)) return;
        event.preventDefault();
        const iframe = document.createElement("iframe");
        iframe.src = `https://www.youtube-nocookie.com/embed/${link.dataset.videoId}?autoplay=1`;
        iframe.title = link.dataset.videoTitle;
        iframe.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
        iframe.allowFullscreen = true;
        iframe.referrerPolicy = "strict-origin-when-cross-origin";
        link.replaceWith(iframe);
        iframe.focus();
    } else if (link.origin === location.origin &&
               link.pathname === location.pathname && link.hash === location.hash) {
        revealArticleAnchor();
    }
});

let closedBeforePrint = [];
window.addEventListener("beforeprint", () => {
    closedBeforePrint = [...document.querySelectorAll(".article-details:not([open])")];
    closedBeforePrint.forEach((details) => { details.open = true; });
});
window.addEventListener("afterprint", () => {
    closedBeforePrint.forEach((details) => { details.open = false; });
    closedBeforePrint = [];
});
