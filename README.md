# Still in development

<center style="font-size:24px;">DISCLAIMER</center>
<center style="font-size:18px;">If you do not understand what this does, do not use it.</center>
<center style="font-size:16px;font-weight:bold;">A closed environment is required. This will transmit.</center>
<center style="font-size:16px;font-weight:bold;">This does not comply with FCC standards.</center>

## Example Usage

### Terminal A
```bash
[user@serverA:~]$ python main.py -d uhd --sample_rate 2e6 --rx_center 434e6 --rx_channel 25000 --tx_center 434e6 --tx_channel 40000
```

### Terminal B
```bash
[user@serverB:~]$ python main.py -d uhd --sample_rate 2e6 --rx_center 434e6 --rx_channel 40000 --tx_center 434e6 --tx_channel 25000
```