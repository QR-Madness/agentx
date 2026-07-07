import { describe, it, expect } from 'vitest';
import {
  encodeConnection,
  parseConnectCode,
  parseConnectFragment,
  buildConnectUrl,
  type ConnectionPayload,
} from './connectionString';

describe('connectionString', () => {
  it('round-trips a full payload', () => {
    const payload: ConnectionPayload = {
      url: 'https://agent.example.com',
      gatewayToken: 's3cr3t-token',
      name: 'Örebro Prod', // unicode → exercises the base64url encoder
    };
    expect(parseConnectCode(encodeConnection(payload))).toEqual(payload);
  });

  it('round-trips a minimal payload and drops empty optionals', () => {
    const decoded = parseConnectCode(encodeConnection({ url: 'http://localhost:12319' }));
    expect(decoded).toEqual({ url: 'http://localhost:12319' });
    expect(decoded?.gatewayToken).toBeUndefined();
    expect(decoded?.name).toBeUndefined();
  });

  it('strips trailing slashes from the url', () => {
    expect(parseConnectCode(encodeConnection({ url: 'https://a.example.com/' }))?.url).toBe(
      'https://a.example.com',
    );
  });

  it('rejects junk / tampered codes', () => {
    expect(parseConnectCode('not-base64!!')).toBeNull();
    expect(parseConnectCode('')).toBeNull();
    expect(parseConnectCode(btoa('{"hello":1}'))).toBeNull(); // no version
  });

  it('rejects non-http(s) urls (no javascript:/file: smuggling)', () => {
    const evil = btoa(JSON.stringify({ v: 1, u: 'javascript:alert(1)' }))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    expect(parseConnectCode(evil)).toBeNull();
  });

  it('parses a connection payload out of a URL fragment', () => {
    const code = encodeConnection({ url: 'https://x.example.com', gatewayToken: 't' });
    expect(parseConnectFragment(`#connect=${code}`)).toEqual({
      url: 'https://x.example.com',
      gatewayToken: 't',
    });
    expect(parseConnectFragment('#connect=')).toBeNull();
    expect(parseConnectFragment('#other=1')).toBeNull();
    expect(parseConnectFragment('')).toBeNull();
  });

  it('builds a shareable link that parses back', () => {
    const payload: ConnectionPayload = { url: 'https://share.example.com', gatewayToken: 'g' };
    const link = buildConnectUrl(payload, 'https://app.example.com');
    expect(link.startsWith('https://app.example.com/#connect=')).toBe(true);
    const hash = link.slice(link.indexOf('#'));
    expect(parseConnectFragment(hash)).toEqual(payload);
  });
});
