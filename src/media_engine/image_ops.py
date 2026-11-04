"""Explicit deterministic RGB image operations, owned code without image-library dependencies."""
from pathlib import Path
from . import png
from .contracts import WorkerError


def validate(params):
    allowed = {"width", "height", "operation", "filter", "x", "y", "background", "opacity", "threshold_low", "threshold_high"}
    if not isinstance(params,dict) or set(params) - allowed or not {"width", "height", "operation"} <= params.keys():
        raise WorkerError(400, "invalid_request", "parameters")
    op = params["operation"]
    if op not in ("resize", "crop", "pad", "composite", "canny"):
        raise WorkerError(400, "unsupported", "operation")
    for key in ("width", "height"):
        if type(params[key]) is not int or not 64 <= params[key] <= 2048 or params[key] % 8:
            raise WorkerError(400, "invalid_request", key)
    for key in ("x", "y"):
        if key in params and (type(params[key]) is not int or not 0 <= params[key] <= 2048):
            raise WorkerError(400, "invalid_request", key)
    if op == "canny":
        low,high=params.get("threshold_low"),params.get("threshold_high")
        if type(low) not in (int,float) or type(high) not in (int,float) or not 0 <= low < high <= 1 or any(k in params for k in ("filter","x","y","background","opacity")):
            raise WorkerError(400,"invalid_request","threshold_low")
    elif "threshold_low" in params or "threshold_high" in params:
        raise WorkerError(400,"invalid_request","threshold_low")
    if op == "resize":
        if params.get("filter") not in ("nearest", "bilinear") or any(k in params for k in ("x", "y", "background", "opacity")):
            raise WorkerError(400, "invalid_request", "filter")
    elif "filter" in params:
        raise WorkerError(400, "invalid_request", "filter")
    if "background" in params and (op != "pad" or not isinstance(params["background"], list) or len(params["background"]) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in params["background"])):
        raise WorkerError(400, "invalid_request", "background")
    if "opacity" in params and (op != "composite" or type(params["opacity"]) not in (int, float) or not 0 <= params["opacity"] <= 1):
        raise WorkerError(400, "invalid_request", "opacity")


def check_dimensions(params, image, overlay=None):
    validate(params)
    iw,ih=image
    ow,oh=params["width"],params["height"]
    x,y=params.get("x",0),params.get("y",0)
    op=params["operation"]
    if op=="canny" and (ow,oh)!=(iw,ih):raise WorkerError(400,"invalid_request","width")
    if op=="crop" and (x+ow>iw or y+oh>ih):raise WorkerError(400,"invalid_request","crop")
    if op=="pad" and (x+iw>ow or y+ih>oh):raise WorkerError(400,"invalid_request","pad")
    if op=="composite" and (overlay is None or (ow,oh)!=(iw,ih) or x+overlay[0]>iw or y+overlay[1]>ih):raise WorkerError(400,"invalid_request","overlay")


def transform(image, params, overlay=None, cancelled=None):
    from .backend import Cancelled
    validate(params)
    iw, ih, source = png.read_rgb(image)
    ow, oh = params["width"], params["height"]
    op = params["operation"]
    x, y = params.get("x", 0), params.get("y", 0)
    if op == "canny":
        check_dimensions(params,(iw,ih))
        from .canny import generate
        return generate(image,params["threshold_low"],params["threshold_high"],cancelled)
    if op == "crop" and (x + ow > iw or y + oh > ih):
        raise WorkerError(400, "invalid_request", "crop")
    if op == "pad" and (x + iw > ow or y + ih > oh):
        raise WorkerError(400, "invalid_request", "pad")
    if op == "composite":
        if overlay is None or (ow, oh) != (iw, ih):
            raise WorkerError(400, "invalid_request", "overlay")
        fw, fh, foreground = png.read_rgb(overlay)
        if x + fw > iw or y + fh > ih:
            raise WorkerError(400, "invalid_request", "overlay")
    target = bytearray(bytes(params.get("background", [0, 0, 0])) * (ow * oh))
    for row in range(oh):
        if cancelled is not None and cancelled.is_set():
            raise Cancelled()
        for col in range(ow):
            dest = (row * ow + col) * 3
            if op == "resize":
                if params["filter"] == "nearest":
                    sx, sy = min(iw - 1, col * iw // ow), min(ih - 1, row * ih // oh)
                    target[dest:dest+3] = source[(sy*iw+sx)*3:(sy*iw+sx)*3+3]
                else:
                    fx, fy = max(0.0, min(iw-1, (col+.5)*iw/ow-.5)), max(0.0, min(ih-1, (row+.5)*ih/oh-.5))
                    sx, sy = int(fx), int(fy); wx, wy = fx-sx, fy-sy
                    for channel in range(3):
                        a=source[(sy*iw+sx)*3+channel];b=source[(sy*iw+min(sx+1,iw-1))*3+channel]
                        c=source[(min(sy+1,ih-1)*iw+sx)*3+channel];d=source[(min(sy+1,ih-1)*iw+min(sx+1,iw-1))*3+channel]
                        target[dest+channel]=int((a*(1-wx)+b*wx)*(1-wy)+(c*(1-wx)+d*wx)*wy+.5)
            elif op == "crop":
                src=((row+y)*iw+col+x)*3;target[dest:dest+3]=source[src:src+3]
            elif op == "pad":
                if x<=col<x+iw and y<=row<y+ih:
                    src=((row-y)*iw+col-x)*3;target[dest:dest+3]=source[src:src+3]
            else:
                target[dest:dest+3]=source[dest:dest+3]
                if x<=col<x+fw and y<=row<y+fh:
                    src=((row-y)*fw+col-x)*3;alpha=params.get("opacity",1.0)
                    for channel in range(3):target[dest+channel]=int(source[dest+channel]*(1-alpha)+foreground[src+channel]*alpha+.5)
    return png.write(ow, oh, bytes(target))


def generate(request, directory, emit, cancelled, image_path):
    paths = image_path if isinstance(image_path, dict) else {"image": image_path}
    emit("stage", {"stage": "transforming"})
    encoded=transform(Path(paths["image"]).read_bytes(),request["parameters"],Path(paths["overlay"]).read_bytes() if paths.get("overlay") else None,cancelled)
    emit("stage", {"stage": "saving"})
    path=Path(directory)/"image.png";path.write_bytes(encoded)
    return [{"output_id":"image","path":path,"mime_type":"image/png","width":request["parameters"]["width"],"height":request["parameters"]["height"]}], {"pixel_operation":request["parameters"]["operation"]}
