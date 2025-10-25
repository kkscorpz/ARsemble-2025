
# --------------------------
# my client data
# --------------------------

data = {
    "gpu": {
        "integrated graphics": {
            "name": "Integrated Graphics (from CPU)",
            "type": "GPU",
            "vram": "Shared system memory",
            "clock": "Varies by CPU",
            "power": "0W (included in CPU power)",
            "slot": "None (integrated)",
            "price": "₱0 (included with CPU)",
            "compatibility": "Works with any compatible motherboard, no additional power required"
        },
        "gtx 750 ti": {
            "name": "NVIDIA GTX 750 Ti",
            "type": "GPU",
            "vram": "2GB GDDR5",
            "clock": "~1085 MHz (Boost)",
            "power": "~60 Watts",
            "slot": "PCIe 3.0 x16",
            "price": "₱4,000",
            "compatibility": "PCIe x16 slot, 300W PSU recommended"
        },
        "rtx 3050": {
            "name": "Gigabyte RTX 3050 EAGLE OC",
            "type": "GPU",
            "vram": "8GB GDDR6",
            "clock": "~1777 MHz (Boost)",
            "power": "~130 Watts",
            "slot": "PCIe 4.0 x16",
            "price": "₱12,000",
            "compatibility": "PCIe x16 slot, 550W PSU, 8-pin power connector"
        },
        "rtx 3060": {
            "name": "MSI RTX 3060",
            "type": "GPU",
            "vram": "12GB GDDR6",
            "clock": "~1777 MHz (Boost)",
            "power": "~170 Watts",
            "slot": "PCIe 4.0 x16",
            "price": "₱16,000",
            "compatibility": "PCIe x16 slot, 550W PSU, 8-pin power connector"
        },
        "rtx 4060": {
            "name": "MSI RTX 4060 GAMING X",
            "type": "GPU",
            "vram": "8GB GDDR6",
            "clock": "~2595 MHz (Boost)",
            "power": "~115 Watts",
            "slot": "PCIe 4.0 x8",
            "price": "₱18,000",
            "compatibility": "PCIe x16 slot, 550W PSU, 8-pin power connector"
        }
    },
    "cpu": {
        "amd ryzen 3 3200g": {
            "name": "AMD Ryzen 3 3200G",
            "type": "CPU",
            "socket": "AM4",
            "cores": "4 Cores / 4 Threads",
            "clock": "3.6 GHz / 4.0 GHz Boost",
            "tdp": "65W",
            "igpu": "Vega 8",
            "price": "₱4,500",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 5 3600": {
            "name": "AMD Ryzen 5 3600",
            "type": "CPU",
            "socket": "AM4",
            "cores": "6 Cores / 12 Threads",
            "clock": "3.6 GHz / 4.2 GHz Boost",
            "tdp": "65W",
            "igpu": "None",
            "price": "₱6,500",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 5 5600g": {
            "name": "AMD Ryzen 5 5600G",
            "type": "CPU",
            "socket": "AM4",
            "cores": "6 Cores / 12 Threads",
            "clock": "3.9 GHz / 4.4 GHz Boost",
            "tdp": "65W",
            "igpu": "Vega 7",
            "price": "₱8,500",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 5 5600x": {
            "name": "AMD Ryzen 5 5600X",
            "type": "CPU",
            "socket": "AM4",
            "cores": "6 Cores / 12 Threads",
            "clock": "3.7 GHz / 4.6 GHz Boost",
            "tdp": "65W",
            "igpu": "None",
            "price": "₱9,000",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 7 5700x": {
            "name": "AMD Ryzen 7 5700X",
            "type": "CPU",
            "socket": "AM4",
            "cores": "8 Cores / 16 Threads",
            "clock": "3.4 GHz / 4.6 GHz Boost",
            "tdp": "65W",
            "igpu": "None",
            "price": "₱12,000",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 7 5800x": {
            "name": "AMD Ryzen 7 5800X",
            "type": "CPU",
            "socket": "AM4",
            "cores": "8 Cores / 16 Threads",
            "clock": "3.8 GHz / 4.7 GHz Boost",
            "tdp": "105W",
            "igpu": "None",
            "price": "₱14,000",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 9 5900x": {
            "name": "AMD Ryzen 9 5900X",
            "type": "CPU",
            "socket": "AM4",
            "cores": "12 Cores / 24 Threads",
            "clock": "3.7 GHz / 4.8 GHz Boost",
            "tdp": "105W",
            "igpu": "None",
            "price": "₱18,000",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "amd ryzen 5 7600": {
            "name": "AMD Ryzen 5 7600",
            "type": "CPU",
            "socket": "AM5",
            "cores": "6 Cores / 12 Threads",
            "clock": "3.8 GHz / 5.1 GHz Boost",
            "tdp": "65W",
            "igpu": "Radeon Graphics",
            "price": "₱13,500",
            "compatibility": "AM5 Motherboards, DDR5 RAM"
        },
        "amd ryzen 7 7700x": {
            "name": "AMD Ryzen 7 7700X",
            "type": "CPU",
            "socket": "AM5",
            "cores": "8 Cores / 16 Threads",
            "clock": "4.5 GHz / 5.4 GHz Boost",
            "tdp": "105W",
            "igpu": "Radeon Graphics",
            "price": "₱18,500",
            "compatibility": "AM5 Motherboards, DDR5 RAM"
        },
        "amd ryzen 9 7900x": {
            "name": "AMD Ryzen 9 7900X",
            "type": "CPU",
            "socket": "AM5",
            "cores": "12 Cores / 24 Threads",
            "clock": "4.7 GHz / 5.6 GHz Boost",
            "tdp": "170W",
            "igpu": "Radeon Graphics",
            "price": "₱25,000",
            "compatibility": "AM5 Motherboards, DDR5 RAM"
        },
        "amd ryzen 9 7950x": {
            "name": "AMD Ryzen 9 7950X",
            "type": "CPU",
            "socket": "AM5",
            "cores": "16 Cores / 32 Threads",
            "clock": "4.5 GHz / 5.7 GHz Boost",
            "tdp": "170W",
            "igpu": "Radeon Graphics",
            "price": "₱32,000",
            "compatibility": "AM5 Motherboards, DDR5 RAM"
        },
        "amd ryzen 5 5600x": {
            "name": "AMD Ryzen 5 5600X",
            "type": "CPU",
            "socket": "AM4",
            "cores": "6 Cores / 12 Threads",
            "clock": "3.7 GHz / 4.6 GHz Boost",
            "tdp": "65W",
            "igpu": "None",
            "price": "₱9,000",
            "compatibility": "AM4 Motherboards, DDR4 RAM"
        },
        "intel core i5 13400": {
            "name": "Intel Core i5 13400",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "10 Cores / 16 Threads",
            "clock": "2.5 GHz / 4.6 GHz Boost",
            "tdp": "65W",
            "igpu": "UHD Graphics 730",
            "price": "₱12,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i3 13100": {
            "name": "Intel Core i3 13100",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "4 Cores / 8 Threads",
            "clock": "3.4 GHz / 4.5 GHz Boost",
            "tdp": "60W",
            "igpu": "UHD Graphics 730",
            "price": "₱6,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i3 14100": {
            "name": "Intel Core i3 14100",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "4 Cores / 8 Threads",
            "clock": "3.5 GHz / 4.7 GHz Boost",
            "tdp": "60W",
            "igpu": "UHD Graphics 730",
            "price": "₱6,500",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i5 13400": {
            "name": "Intel Core i5 13400",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "10 Cores / 16 Threads",
            "clock": "2.5 GHz / 4.6 GHz Boost",
            "tdp": "65W",
            "igpu": "UHD Graphics 730",
            "price": "₱12,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i5 14500": {
            "name": "Intel Core i5 14500",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "14 Cores / 20 Threads",
            "clock": "2.6 GHz / 4.8 GHz Boost",
            "tdp": "65W",
            "igpu": "UHD Graphics 730",
            "price": "₱13,500",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i5 14600k": {
            "name": "Intel Core i5 14600K",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "14 Cores / 20 Threads",
            "clock": "3.5 GHz / 5.3 GHz Boost",
            "tdp": "125W",
            "igpu": "UHD Graphics 770",
            "price": "₱16,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i7 13700k": {
            "name": "Intel Core i7 13700K",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "16 Cores / 24 Threads",
            "clock": "3.4 GHz / 5.4 GHz Boost",
            "tdp": "125W",
            "igpu": "UHD Graphics 770",
            "price": "₱22,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i7 14700k": {
            "name": "Intel Core i7 14700K",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "20 Cores / 28 Threads",
            "clock": "3.4 GHz / 5.6 GHz Boost",
            "tdp": "125W",
            "igpu": "UHD Graphics 770",
            "price": "₱24,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        },
        "intel core i9 14900k": {
            "name": "Intel Core i9 14900K",
            "type": "CPU",
            "socket": "LGA1700",
            "cores": "24 Cores / 32 Threads",
            "clock": "3.2 GHz / 6.0 GHz Boost",
            "tdp": "125W",
            "igpu": "UHD Graphics 770",
            "price": "₱32,000",
            "compatibility": "LGA1700 Motherboards, DDR4/DDR5"
        }
    },
    "motherboard": {
        "gigabyte h610m k ddr4": {
            "name": "GIGABYTE H610M K DDR4",
            "type": "Motherboard",
            "socket": "LGA1700",
            "form_factor": "mATX",
            "ram_slots": 2,
            "max_ram": "64GB",
            "ram_type": "DDR4",
                        "nvme_slots": 1,
                        "sata_ports": 4,
                        "price": "₱4,500",
                        "compatibility": "LGA1700 CPUs, DDR4 RAM, PCIe 4.0"
        },
        "msi pro h610m s ddr4": {
            "name": "MSI Pro H610M S DDR4",
            "type": "Motherboard",
            "socket": "LGA1700",
            "form_factor": "mATX",
            "ram_slots": 2,
            "max_ram": "64GB",
            "ram_type": "DDR4",
                        "nvme_slots": 1,
                        "sata_ports": 4,
                        "price": "₱4,800",
                        "compatibility": "LGA1700 CPUs, DDR4 RAM, PCIe 4.0"
        },
        "msi b450m-a pro max ii": {
            "name": "MSI B450M-A PRO MAX II",
            "type": "Motherboard",
            "socket": "AM4",
            "form_factor": "mATX",
            "ram_slots": 2,
            "max_ram": "64GB",
            "ram_type": "DDR4",
                        "nvme_slots": 1,
                        "sata_ports": 4,
                        "price": "₱4,200",
                        "compatibility": "AM4 CPUs, DDR4 RAM, PCIe 3.0"
        },
        "ramsta rs-b450mp": {
            "name": "RAMSTA RS-B450MP",
            "type": "Motherboard",
            "socket": "AM4",
            "form_factor": "mATX",
            "ram_slots": 2,
            "max_ram": "64GB",
            "ram_type": "DDR4",
                        "nvme_slots": 1,
                        "sata_ports": 4,
                        "price": "₱3,800",
                        "compatibility": "AM4 CPUs, DDR4 RAM, PCIe 3.0"
        },
        "asus tuf gaming b550-plus": {
            "name": "ASUS TUF GAMING B550-PLUS",
            "type": "Motherboard",
            "socket": "AM4",
            "form_factor": "ATX",
            "ram_slots": 4,
            "max_ram": "128GB",
            "ram_type": "DDR4",
                        "nvme_slots": 2,
                        "sata_ports": 6,
                        "price": "₱7,800",
                        "compatibility": "AM4 CPUs, DDR4 RAM, PCIe 4.0"
        },
        "asus prime b650-plus": {
            "name": "ASUS PRIME B650-PLUS",
            "type": "Motherboard",
            "socket": "AM5",
            "form_factor": "ATX",
            "ram_slots": 4,
            "max_ram": "128GB",
            "ram_type": "DDR5",
                        "nvme_slots": 3,
                        "sata_ports": 4,
                        "price": "₱9,500",
                        "compatibility": "AM5 CPUs, DDR5 RAM, PCIe 4.0"
        },
        "msi pro x670-p wifi": {
            "name": "MSI PRO X670-P WIFI",
            "type": "Motherboard",
            "socket": "AM5",
            "form_factor": "ATX",
            "ram_slots": 4,
            "max_ram": "128GB",
            "ram_type": "DDR5",
                        "nvme_slots": 4,
                        "sata_ports": 6,
                        "price": "₱14,000",
                        "compatibility": "AM5 CPUs, DDR5 RAM, PCIe 5.0"
        },
        "msi mpg z790 carbon wifi": {
            "name": "MSI MPG Z790 CARBON WIFI",
            "type": "Motherboard",
            "socket": "LGA1700",
            "form_factor": "ATX",
            "ram_slots": 4,
            "max_ram": "128GB",
            "ram_type": "DDR5",
                        "nvme_slots": 5,
                        "sata_ports": 6,
                        "price": "₱18,500",
                        "compatibility": "LGA1700 CPUs, DDR5 RAM, PCIe 5.0"
        }
    },
    "ram": {
        "kingston fury beast ddr4 8gb": {
            "name": "Kingston FURY Beast DDR4 8GB",
            "type": "RAM",
            "capacity": "8GB",
            "speed": "3200MHz",
            "ram_type": "DDR4",
                        "price": "₱1,500",
                        "compatibility": "DDR4 Motherboards (AM4, LGA1700 DDR4)"
        },
        "kingston fury beast ddr4 16gb": {
            "name": "Kingston FURY Beast DDR4 16GB",
            "type": "RAM",
            "capacity": "16GB",
            "speed": "3200MHz",
            "ram_type": "DDR4",
                        "price": "₱3,000",
                        "compatibility": "DDR4 Motherboards (AM4, LGA1700 DDR4)"
        },
        "hkcmemory hu40 ddr4 16gb": {
            "name": "HKCMEMORY HU40 DDR4 16GB",
            "type": "RAM",
            "capacity": "16GB",
            "speed": "3200MHz",
            "ram_type": "DDR4",
                        "price": "₱2,200",
                        "compatibility": "DDR4 Motherboards (AM4, LGA1700 DDR4)"
        },
        "kingston fury beast ddr4 32gb": {
            "name": "Kingston FURY Beast DDR4 32GB",
            "type": "RAM",
            "capacity": "32GB",
            "speed": "3200MHz",
            "ram_type": "DDR4",
                        "price": "₱3,800",
                        "compatibility": "DDR4 Motherboards (AM4, LGA1700 DDR4)"
        },
        "kingston fury beast ddr5 8gb": {
            "name": "Kingston FURY Beast DDR5 8GB",
            "type": "RAM",
            "capacity": "8GB",
            "speed": "4800MHz",
            "ram_type": "DDR5",
                        "price": "₱2,000",
                        "compatibility": "DDR5 Motherboards (AM5, LGA1700 DDR5)"
        },
        "kingston fury beast ddr5 16gb": {
            "name": "Kingston FURY Beast DDR5 16GB",
            "type": "RAM",
            "capacity": "16GB",
            "speed": "4800MHz",
            "ram_type": "DDR5",
                        "price": "₱3,000",
                        "compatibility": "DDR5 Motherboards (AM5, LGA1700 DDR5)"
        },
        "corsair vengeance ddr5 32gb": {
            "name": "Corsair Vengeance DDR5 32GB",
            "type": "RAM",
            "capacity": "32GB",
            "speed": "5200MHz",
            "ram_type": "DDR5",
                        "price": "₱5,500",
                        "compatibility": "DDR5 Motherboards (AM5, LGA1700 DDR5)"
        }
    },
    "storage": {
        "seagate video 3.5\" hdd 500gb": {
            "name": "Seagate Video 3.5\" HDD 500GB",
            "type": "HDD",
            "capacity": "500GB",
            "interface": "SATA 6Gb/s",
            "price": "₱1,200",
            "compatibility": "Any motherboard with SATA port"
        },
        "seagate video 3.5\" hdd 1tb": {
            "name": "Seagate Video 3.5\" HDD 1TB",
            "type": "HDD",
            "capacity": "1TB",
            "interface": "SATA 6Gb/s",
            "price": "₱1,800",
            "compatibility": "Any motherboard with SATA port"
        },
        "ramsta s800 128gb": {
            "name": "Ramsta S800 128GB SSD",
            "type": "SATA SSD",
            "capacity": "128GB",
            "interface": "SATA 6Gb/s",
            "price": "₱800",
            "compatibility": "Any motherboard with SATA port"
        },
        "ramsta s800 256gb": {
            "name": "Ramsta S800 256GB SSD",
            "type": "SATA SSD",
            "capacity": "256GB",
            "interface": "SATA 6Gb/s",
            "price": "₱1,200",
            "compatibility": "Any motherboard with SATA port"
        },
        "ramsta s800 512gb": {
            "name": "Ramsta S800 512GB SSD",
            "type": "SATA SSD",
            "capacity": "512GB",
            "interface": "SATA 6Gb/s",
            "price": "₱1,800",
            "compatibility": "Any motherboard with SATA port"
        },
        "ramsta s800 1tb": {
            "name": "Ramsta S800 1TB SSD",
            "type": "SATA SSD",
            "capacity": "1TB",
            "interface": "SATA 6Gb/s",
            "price": "₱2,800",
            "compatibility": "Any motherboard with SATA port"
        },
        "ramsta s800 2tb": {
            "name": "Ramsta S800 2TB SSD",
            "type": "SATA SSD",
            "capacity": "2TB",
            "interface": "SATA 6Gb/s",
            "price": "₱4,500",
            "compatibility": "Any motherboard with SATA port"
        },
        "crucial mx500 500gb": {
            "name": "Crucial MX500 500GB SSD",
            "type": "SATA SSD",
            "capacity": "500GB",
            "interface": "SATA 6Gb/s",
            "price": "₱2,200",
            "compatibility": "Any motherboard with SATA port"
        },
        "samsung 970 evo plus 250gb": {
            "name": "Samsung 970 EVO Plus 250GB",
            "type": "NVMe SSD",
            "capacity": "250GB",
            "interface": "PCIe 3.0 x4",
            "price": "₱1,800",
            "compatibility": "Motherboard with M.2 NVMe slot"
        },
        "samsung 970 evo plus 500gb": {
            "name": "Samsung 970 EVO Plus 500GB",
            "type": "NVMe SSD",
            "capacity": "500GB",
            "interface": "PCIe 3.0 x4",
            "price": "₱2,500",
            "compatibility": "Motherboard with M.2 NVMe slot"
        },
        "samsung 970 evo plus 1tb": {
            "name": "Samsung 970 EVO Plus 1TB",
            "type": "NVMe SSD",
            "capacity": "1TB",
            "interface": "PCIe 3.0 x4",
            "price": "₱4,000",
            "compatibility": "Motherboard with M.2 NVMe slot"
        },
        "samsung 970 evo plus 2tb": {
            "name": "Samsung 970 EVO Plus 2TB",
            "type": "NVMe SSD",
            "capacity": "2TB",
            "interface": "PCIe 3.0 x4",
            "price": "₱7,000",
            "compatibility": "Motherboard with M.2 NVMe slot"
        }
    },
    "psu": {
        "inplay ak400": {
            "name": "InPlay AK400",
            "type": "PSU",
            "wattage": "400W",
            "efficiency": "80+",
            "price": "₱1,200",
            "compatibility": "Basic builds, low-power components"
        },
        "inplay gs 550": {
            "name": "InPlay GS 550",
            "type": "PSU",
            "wattage": "550W",
            "efficiency": "80+ Bronze",
            "price": "₱1,800",
            "compatibility": "Mid-range builds, single GPU systems"
        },
        "corsair cx650": {
            "name": "Corsair CX650",
            "type": "PSU",
            "wattage": "650W",
            "efficiency": "80+ Bronze",
            "price": "₱3,500",
            "compatibility": "Gaming builds, most single GPU configurations"
        },
        "inplay gs 750": {
            "name": "InPlay GS 750",
            "type": "PSU",
            "wattage": "750W",
            "efficiency": "80+ Bronze",
            "price": "₱2,500",
            "compatibility": "High-end builds, powerful GPUs"
        },
        "cooler master mwe white 750w": {
            "name": "Cooler Master MWE White 750W",
            "type": "PSU",
            "wattage": "750W",
            "efficiency": "80+ White",
            "price": "₱3,800",
            "compatibility": "High-end gaming builds, multiple components"
        },
        "corsair rm850x 850w": {
            "name": "Corsair RM850x 850W",
            "type": "PSU",
            "wattage": "850W",
            "efficiency": "80+ Gold",
            "price": "₱6,500",
            "compatibility": "Premium builds, high-end GPUs, overclocking"
        }
    },
    "cpu_cooler": {
        "fantech polar lc240": {
            "name": "Fantech Polar LC240",
            "type": "CPU Cooler",
            "cooler_type": "Liquid Cooler",
            "size": "240mm",
            "socket": "AM4, AM5, LGA1700, LGA1200, LGA1151",
            "price": "₱2,800",
            "compatibility": "Most modern CPU sockets"
        },
        "inplay seaview 240 pro": {
            "name": "Inplay Seaview 240 Pro",
            "type": "CPU Cooler",
            "cooler_type": "Liquid Cooler",
            "size": "240mm",
            "socket": "AM4, AM5, LGA1700, LGA1200",
            "price": "₱2,200",
            "compatibility": "Modern CPU sockets"
        },
        "inplay seaview 360 pro": {
            "name": "Inplay Seaview 360 Pro",
            "type": "CPU Cooler",
            "cooler_type": "Liquid Cooler",
            "size": "360mm",
            "socket": "AM4, AM5, LGA1700, LGA1200",
            "price": "₱2,800",
            "compatibility": "Modern CPU sockets"
        },
        "inplay s20": {
            "name": "Inplay S20",
            "type": "CPU Cooler",
            "cooler_type": "Air Cooler",
            "size": "120mm",
            "socket": "AM4, LGA1700, LGA1200",
            "price": "₱600",
            "compatibility": "Basic cooling for low to mid-range CPUs"
        },
        "inplay s40": {
            "name": "Inplay S40",
            "type": "CPU Cooler",
            "cooler_type": "Air Cooler",
            "size": "120mm",
            "socket": "AM4, AM5, LGA1700, LGA1200",
            "price": "₱800",
            "compatibility": "Mid-range CPUs, good value cooling"
        },
        "cooler master hyper 212 black edition": {
            "name": "Cooler Master Hyper 212 Black Edition",
            "type": "CPU Cooler",
            "cooler_type": "Air Cooler",
            "size": "120mm",
            "socket": "AM4, AM5, LGA1700, LGA1200, LGA1151",
            "price": "₱2,000",
            "compatibility": "Excellent air cooling for mid to high-end CPUs"
        },
        "deepcool ls720 se 360": {
            "name": "DeepCool LS720 SE 360",
            "type": "CPU Cooler",
            "cooler_type": "Liquid Cooler",
            "size": "360mm",
            "socket": "AM4, AM5, LGA1700, LGA1200",
            "price": "₱4,500",
            "compatibility": "High-performance cooling"
        },
        "cooler master masterliquid ml360r rgb": {
            "name": "Cooler Master MasterLiquid ML360R RGB",
            "type": "CPU Cooler",
            "cooler_type": "Liquid Cooler",
            "size": "360mm",
            "socket": "AM4, AM5, LGA1700, LGA2066, LGA1200",
            "price": "₱5,500",
            "compatibility": "Premium cooling solution, high-end CPUs"
        }
    }
}
