"use strict";
const electron = require("electron");
const preload = require("@electron-toolkit/preload");
const api = {
  // Legacy IPC methods - now return promises that will be handled by BackendAPI
  // Components should migrate to using BackendAPI directly for better real-time features
  getBackendData: (route = "time") => {
    console.warn("api.getBackendData is deprecated. Use BackendAPI.getBackendData() directly in components");
    return Promise.reject(new Error("Use BackendAPI service instead of preload IPC"));
  },
  getTimeData: () => {
    console.warn("api.getTimeData is deprecated. Use BackendAPI.getTimeData() directly in components");
    return Promise.reject(new Error("Use BackendAPI service instead of preload IPC"));
  },
  getRandomNumber: () => {
    console.warn("api.getRandomNumber is deprecated. Use BackendAPI.getRandomNumber() directly in components");
    return Promise.reject(new Error("Use BackendAPI service instead of preload IPC"));
  },
  getSystemInfo: () => {
    console.warn("api.getSystemInfo is deprecated. Use BackendAPI.getSystemInfo() directly in components");
    return Promise.reject(new Error("Use BackendAPI service instead of preload IPC"));
  },
  getWeatherData: () => {
    console.warn("api.getWeatherData is deprecated. Use BackendAPI.getWeatherData() directly in components");
    return Promise.reject(new Error("Use BackendAPI service instead of preload IPC"));
  },
  processAudio: (audioData, filename, processType) => {
    console.warn("api.processAudio is deprecated. Use BackendAPI.processAudio() directly in components");
    return Promise.reject(new Error("Use BackendAPI service instead of preload IPC"));
  }
};
if (process.contextIsolated) {
  try {
    electron.contextBridge.exposeInMainWorld("electron", preload.electronAPI);
    electron.contextBridge.exposeInMainWorld("api", api);
  } catch (error) {
  }
} else {
  window.electron = preload.electronAPI;
  window.api = api;
}
