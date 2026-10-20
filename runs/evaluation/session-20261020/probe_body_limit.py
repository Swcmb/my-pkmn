# -*- coding: utf-8 -*-
"""探测平台 nginx 的请求体上限（client_max_body_size）。

原理：带 `Expect: 100-continue` 时，服务器在**读取 body 之前**就根据 Content-Length 决定是否 413。
所以我们声明一个长度、只在收到 100 Continue 时立刻关闭连接（不传任何 body），
用二分确定阈值。全程不产生任何平台写入：413 由 nginx 拦截，其余 4xx 来自应用层的鉴权/参数校验。
"""
import socket
import sys

HOST, PORT, PATH = "115.159.47.81", 80, "/api/submissions"


def probe(content_length):
    req = (
        f"POST {PATH} HTTP/1.1\r\n"
        f"Host: {HOST}\r\n"
        f"Content-Type: multipart/form-data; boundary=----probe\r\n"
        f"Content-Length: {content_length}\r\n"
        f"Expect: 100-continue\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    s = socket.create_connection((HOST, PORT), timeout=15)
    try:
        s.sendall(req)
        head = s.recv(4096).decode(errors="replace")
    finally:
        s.close()          # 收到判定后立即断开，不发送 body
    status = head.split(" ", 2)[1] if head.startswith("HTTP/") and len(head.split(" ", 2)) > 1 else "?"
    interim = head.splitlines()[0] if head.startswith("HTTP/1.1 100") else ""
    return status, interim or head.splitlines()[0][:60]


if __name__ == "__main__":
    mb = 1024 * 1024
    sizes = [1 * mb, 2 * mb, 5 * mb, 10 * mb, 20 * mb, 32 * mb, 50 * mb, 64 * mb, 72 * mb, 100 * mb, 128 * mb]
    print(f"{'声明长度':>10}  {'响应':<46}")
    for n in sizes:
        st, info = probe(n)
        print(f"{n//mb:>8} MB  {st:>4}  {info[:44]}")
