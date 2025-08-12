from werkzeug.utils import secure_filename

def is_valid_xlsx_file(file) -> bool:
    from app import logger
    if not file or not getattr(file, "filename", None):
        return False

    try:
        filename = secure_filename(file.filename.lower())
        if not filename.endswith(".xlsx"):
            logger.warning(f"Invalid extension: {filename}")
            return False

        allowed_mimes = {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/octet-stream",
        }
        if getattr(file, "content_type", "") not in allowed_mimes:
            logger.info(f"Non-standard MIME: {file.content_type} (continuing)")

        try:
            pos = file.tell()
        except Exception:
            pos = None

        try:
            file.seek(0)
            header = file.read(4)
        finally:
            if pos is not None:
                file.seek(pos)

        if not header or header[:2] != b"PK":
            logger.warning("Invalid file signature")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False