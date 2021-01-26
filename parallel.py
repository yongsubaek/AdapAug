import os
import time
import random
import concurrent
import asyncio
import subprocess
import argparse


def run_parallel(commands, n_worker=1):
    assert commands and type(commands) == list

    def func():
        while True:
            try:
                time.sleep(random.random() * n_worker)
                if not commands:
                    break

                command = commands.pop()
                os.system(command)
                # for cmd in command:
                    # os.system(cmd)
            except KeyboardInterrupt:
                break

    async def main():
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_worker) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(executor, func) for _ in range(n_worker)
            ]

            for response in await asyncio.gather(*futures):
                pass

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()

    with open('./schedule.txt') as f:
        commands = f.read().split('\n')

    # with open('./schedule.txt') as f:
    #     commands = f.read().split('\n\n')
    # command_lists = [command.split('\n') for command in commands]

    # suffix_list = []
    # cnt = 0
    # for d in datasets:
    #     for s in seeds:
    #         if args.comment == '':
    #             comment = '{}'.format(str(s))
    #         else:
    #             comment = args.comment + '_{}'.format(str(s))
    #         suffix = ' --dataset {} --seed {} --device {} --comment {}'.format(
    #             d, str(s), str(devices[cnt % len(devices)]), comment,
    #         )
    #         suffix_list.append(suffix)
    #         cnt += 1

    # command_lists = []
    # for command in commands:
    #     command_lists.append([command + suffix for suffix in suffix_list])

    # run_parallel(sum(command_lists,[]), n_worker=2*len(devices))
    run_parallel(commands, n_worker=8)
