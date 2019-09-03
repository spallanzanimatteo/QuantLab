{
  "description": "randomDescription",
  "indiv": {
    "net": {
        "class": "VGG",
        "params": {
        "quantAct": false, 
        "quantWeights": true,
        "weightInqBits": 2, 
        "weightInqSchedule": {
          "10": 0.9,
          "100": 0.9,
          "200": 0.9,
          "300": 0.9,
          "400": 0.9,
          "500": 0.9,
          "600": 0.9,
          "700": 0.9,
          "800": 0.9,
          "900": 0.9
        }
      },
      "pretrained": {
          "file": {
              "exp_id": 34, 
              "epoch": 1300
          }
      }
    },
    "loss_function": {
      "class": "CrossEntropyLoss",
      "params": {}
    }
  },
  "treat": {
    "thermostat": {
      "params": {
        "noise_schemes": {},
        "bindings": []
      }
    },
    "optimizer": {
      "class": "Adam",
      "params": {"lr": 1e-3}
    },
    "lr_scheduler": {
        "class": "HandScheduler",
        "params": {
          "schedule": {
               "11": 1.0,
              "101": 1.0,
              "201": 1.0,
              "301": 1.0,
              "401": 1.0,
              "501": 1.0,
              "601": 1.0,
              "701": 1.0,
              "801": 1.0,
              "901": 1.0,
    
              "150": 0.1,
              "250": 0.1,
              "350": 0.1,
              "450": 0.1,
              "550": 0.1,
              "650": 0.1,
              "750": 0.1,
              "850": 0.1,
              "950": 0.1,
    
             "1000": 0.01
          }
        }
    },
    "data": {
      "augment": true,
      "valid_fraction": 0.1,
      "bs_train": 256,
      "bs_valid": 64
    },
    "max_epoch": 1100
  },
  "protocol": {
    "update_metric": "valid_metric",
    "metric_period": 1
  }
}
