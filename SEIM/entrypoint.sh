/usr/sbin/nginx &
echo "Start nginx.."

/usr/bin/suricata $@ &
echo "Start suricata..."

# Wait for any process to exit
wait
echo $?
# Exit with status of process that exited first
exit $?
