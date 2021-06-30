def is_iree_enabled():
    try:
        import iree.runtime
        import iree.compiler
    except:
        return False
    return True
