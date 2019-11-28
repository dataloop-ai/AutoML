from threading import Thread


def _join_threads(threads):
    # takes dictionary of threads and joins all to main thread
    for id_hash, thread in threads.items():
        thread.join()


class ThreadManager:

    def __init__(self):
        self.ongoing_results = {}
        self.ongoing_threads = {}

    def new_thread(self, target, inputs, trial_id):
        name = 'trial_' + trial_id
        args = (inputs, trial_id, self.ongoing_results)
        self.ongoing_threads[name] = Thread(target=target, args=args, name=name,
                                            daemon=True)
        self.ongoing_threads[name].start()

    def wait(self):
        _join_threads(self.ongoing_threads)

    @property
    def results(self):
        return self.ongoing_results
