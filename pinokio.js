const path = require("path");
module.exports = {
  version: "3.7",
  title: "PromptForge",
  description: "A local prompt-writing assistant for Ideogram 4, MiniMax H3 video, and plain-text image models, powered by a local LLM (llama-server).",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("env");
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
    };
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }];
    } else if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-arrows-rotate",
        text: "Updating",
        href: "update.js",
      }];
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js");
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open PromptForge",
            href: local.url,
            popout: true,
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }];
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }];
        }
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        }, {
          icon: "fa-solid fa-arrows-rotate",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }];
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }];
    }
  },
};
