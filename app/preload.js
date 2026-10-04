'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gvision', {
  onMessage: (cb) => ipcRenderer.on('gvision:message', (_e, msg) => cb(msg)),
  onConnection: (cb) => ipcRenderer.on('gvision:connection', (_e, state) => cb(state)),
  getConnection: () => ipcRenderer.invoke('gvision:connection'),
  send: (msg) => ipcRenderer.invoke('gvision:send', msg),
  onServices: (cb) => ipcRenderer.on('gvision:services', (_e, state) => cb(state)),
  getServices: () => ipcRenderer.invoke('gvision:services'),
  restartService: (name) => ipcRenderer.invoke('gvision:restart-service', name),
  stopService: (name) => ipcRenderer.invoke('gvision:stop-service', name),
  openLogs: () => ipcRenderer.invoke('gvision:open-logs'),
  update: () => ipcRenderer.invoke('gvision:update'),
  onExchange: (cb) => ipcRenderer.on('gvision:exchange', (_e, entry) => cb(entry)),
  getLog: () => ipcRenderer.invoke('gvision:get-log'),
  clearLog: () => ipcRenderer.invoke('gvision:clear-log'),
  getSettings: () => ipcRenderer.invoke('gvision:get-settings'),
  saveSettings: (changes) => ipcRenderer.invoke('gvision:save-settings', changes),
  getVisionModels: () => ipcRenderer.invoke('gvision:vision-models'),
  useVisionModel: (id) => ipcRenderer.invoke('gvision:use-vision-model', id),
  cancelVisionDownload: () => ipcRenderer.invoke('gvision:cancel-vision-download'),
  onVisionDownload: (cb) => ipcRenderer.on('gvision:vision-download', (_e, p) => cb(p)),
});
