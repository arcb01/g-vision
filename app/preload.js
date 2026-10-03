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
});
