echo "usage: source mv_shell_to_fio_limit.sh"
echo $$ | sudo tee /sys/fs/cgroup/fio_limit/cgroup.procs
cat /proc/$$/cgroup