import stopit
import sys
# ...
with stopit.ThreadingTimeout(100) as to_ctx_mrg:
    assert to_ctx_mrg.state == to_ctx_mrg.EXECUTING
    # Something potentially very long but which
    # ...
    # while True:
    print("ai")

# OK, let's check what happened
if to_ctx_mrg.state == to_ctx_mrg.EXECUTED:
    # All's fine, everything was executed within 10 seconds
    print("not possible")
elif to_ctx_mrg.state == to_ctx_mrg.EXECUTING:
    # Hmm, that's not possible outside the block
    print ("?")
elif to_ctx_mrg.state == to_ctx_mrg.TIMED_OUT:
    # Eeek the 10 seconds timeout occurred while executing the block
    print ("timeout")
    sys.exit(0)
elif to_ctx_mrg.state == to_ctx_mrg.INTERRUPTED:
    # Oh you raised specifically the TimeoutException in the block
    print ("INTERRUPTED")
elif to_ctx_mrg.state == to_ctx_mrg.CANCELED:
    # Oh you called to_ctx_mgr.cancel() method within the block but it
    # executed till the end
    print ("CANCELED")