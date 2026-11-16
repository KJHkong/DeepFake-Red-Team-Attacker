"""Bounded byte-only Thrift NISP client; no service lifecycle or image paths.

Deployment identity is server configuration verified by the deployment caller,
not self-reported by the original IDL. Same client runs inside the CPU container.
Transport: https://thrift.apache.org/tutorial/py.html
"""
from contextlib import contextmanager
import hashlib
import math
from time import perf_counter
from uuid import uuid4

LABELS = ('frontal', 'left', 'right', 'mouth', 'blink')


class NispThriftClient:
    def __init__(self, *, host='127.0.0.1', port=9002, timeout_ms=10000,
                 identity=None, connection_factory=None, stubs=None):
        if host not in ('127.0.0.1', 'localhost') or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Explicit local endpoint required')
        if type(timeout_ms) is not int or not 0 < timeout_ms <= 30000:
            raise ValueError('Bounded NISP deadline required')
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.identity = dict(identity or {})
        self._factory = connection_factory or self._connection
        self._stubs = stubs

    def stubs(self):
        if self._stubs is None:
            raise RuntimeError("NISP deployment stubs are not distributed; inject authorized stubs explicitly")
        return self._stubs

    @contextmanager
    def _connection(self):
        from thrift.transport import TSocket, TTransport
        from thrift.protocol import TBinaryProtocol
        service, _, _ = self.stubs()
        sock = TSocket.TSocket(self.host, self.port)
        sock.setTimeout(self.timeout_ms)
        transport = TTransport.TBufferedTransport(sock)
        try:
            transport.open()
            yield service.Client(TBinaryProtocol.TBinaryProtocol(transport))
        finally:
            transport.close()

    def analyze(self, artifact, data):
        start = perf_counter()
        out = dict(status='error', source_sha256=artifact.sha256, artifact_id=artifact.artifact_id,
                   identity=self.identity, raw_response=None, scores={}, error=None, rpc_status='not_executed',
                   semantics='rounded class scores; not calibrated probabilities or continuous measurements',
                   capabilities={k: 'unsupported' for k in ['face_count', 'yaw_degrees', 'pitch_degrees', 'roll_degrees', 'eye_ratios', 'lip_ratio']})
        try:
            if not isinstance(data, bytes) or not 0 < len(data) <= 20*1024**2 or hashlib.sha256(data).hexdigest() != artifact.sha256:
                raise ValueError('NISP input bytes changed')
            _, params, unit = self.stubs()
            out['rpc_status']='attempted'
            with self._factory() as client:
                response = client.infer(params(uuid=uuid4().hex, params=[unit(label=k, data=data) for k in LABELS]))
            out['rpc_status']='completed'
            if not isinstance(response, dict) or set(response) != set(LABELS):
                raise ValueError('Unexpected NISP map labels')
            raw, scores, errors = {}, {}, {}
            no_face = False
            for key, value in response.items():
                row = dict(status=value.status, cost=value.cost, message=value.message,
                           unit=[dict(label=u.label, score=u.score) for u in value.unit])
                raw[key] = row
                if type(value.status) is not int or value.status not in (200, 500) or type(value.cost) is not int or value.cost < 0:
                    raise ValueError('Invalid NISP status/cost')
                message = value.message or ''
                if not isinstance(message, str) or len(message) > 4096:
                    raise ValueError('Invalid NISP message')
                no_face |= 'none face' in message.lower() or 'no face' in message.lower()
                if value.status != 200 or message or len(value.unit) != 1:
                    errors[key] = 'service_error_or_unusable_unit'
                    continue
                u = value.unit[0]
                if u.label != key or type(u.score) not in (float, int) or not math.isfinite(u.score) or not 0 <= u.score <= 1:
                    raise ValueError('Invalid NISP score')
                scores[key] = float(u.score)
            out.update(raw_response=raw, scores=scores, service_errors=errors,
                       status='no_face' if no_face else 'error' if errors else 'success')
        except Exception as error:
            from thrift.transport.TTransport import TTransportException
            timeout = isinstance(error, (TimeoutError,)) or (isinstance(error, TTransportException) and error.type == TTransportException.TIMED_OUT)
            if out['rpc_status']=='attempted':out['rpc_status']='timeout' if timeout else 'failed'
            out['error'] = dict(code='nisp_timeout' if timeout else 'nisp_transport_or_response_error', type=type(error).__name__)
        out['elapsed_ms'] = (perf_counter()-start)*1000
        return out
