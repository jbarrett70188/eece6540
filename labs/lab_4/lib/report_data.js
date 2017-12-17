var loopsJSON={
  "columns":["Pipelined", "II", "Bottleneck"]
  , "functions":
  [
    {
      "name":"Block4"
      , "data":
      ["Yes", "1", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"fft1d.cl"
            , "line":55
            , "level":0
          }
        ]
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"fft_8.cl"
            , "line":295
            , "level":1
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"fft_8.cl"
            , "line":337
            , "level":1
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
    , {
      "name":"Block9"
      , "data":
      ["Yes", "1", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"filter.cl"
            , "line":51
            , "level":0
          }
        ]
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"filter.cl"
            , "line":58
            , "level":1
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"filter.cl"
            , "line":62
            , "level":1
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"filter.cl"
            , "line":69
            , "level":1
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"filter.cl"
            , "line":83
            , "level":1
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
    , {
      "name":"Fully unrolled loop"
      , "data":
      ["n/a", "n/a", "n/a"]
      , "debug":
      [
        [
          {
            "filename":"filter.cl"
            , "line":86
            , "level":2
          }
        ]
      ]
      , "details":
      [
        "Unrolled by #pragma unroll"
      ]
    }
  ]
}
;var mavJSON={
  "nodes":
  [
    {
      "type":"kernel"
      , "id":10
      , "name":"data_in"
      , "file":""
      , "line":"0"
      , "children":[
        {
          "type":"bb"
          , "id":3
          , "name":"Block0"
          , "file":""
          , "line":"0"
          , "children":[
            {
              "type":"inst"
              , "id":4
              , "name":"Load"
              , "file":"1"
              , "line":"74"
              , "details":
              {
                "Width":"256 bits"
                , "Type":"Burst-coalesced"
                , "Stall-free":"No"
                , "Start-Cycle":"2"
                , "Latency":"147"
              }
            }
            , {
              "type":"inst"
              , "id":5
              , "name":"Channel Write"
              , "file":"1"
              , "line":"75"
              , "details":
              {
                "Width":"256 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"149"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":7
              , "name":"end"
              , "file":"1"
              , "line":"76"
              , "details":
              {
                "Start-Cycle":"150"
                , "Latency":"1"
                , "Additional Info":"Exit from a basic block. Control flow branches at this node to one or more merge nodes. There is no control branching between merge and branch node for the same basic block."
              }
            }
            , {
              "type":"inst"
              , "id":8
              , "name":"begin"
              , "file":""
              , "line":""
              , "details":
              {
                "Start-Cycle":"0"
                , "Latency":"1"
                , "Additional Info":"Entrance to a basic block. Control flow comes to this node from one or more branch nodes, unless it's the very first merge node in a kernel. There is no control branching between merge and branch node within the same basic block."
              }
            }
          ]
          , "details":
          {
            "Latency":"151"
          }
        }
      ]
    }
    , {
      "type":"kernel"
      , "id":23
      , "name":"data_out"
      , "file":""
      , "line":"0"
      , "children":[
        {
          "type":"bb"
          , "id":16
          , "name":"Block1"
          , "file":""
          , "line":"0"
          , "children":[
            {
              "type":"inst"
              , "id":17
              , "name":"Channel Read"
              , "file":"1"
              , "line":"79"
              , "details":
              {
                "Width":"512 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"1"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":19
              , "name":"Store"
              , "file":"1"
              , "line":"82"
              , "details":
              {
                "Width":"256 bits"
                , "Type":"Burst-coalesced"
                , "Stall-free":"No"
                , "Start-Cycle":"19"
                , "Latency":"4"
              }
            }
            , {
              "type":"inst"
              , "id":20
              , "name":"end"
              , "file":"1"
              , "line":"83"
              , "details":
              {
                "Start-Cycle":"23"
                , "Latency":"1"
                , "Additional Info":"Exit from a basic block. Control flow branches at this node to one or more merge nodes. There is no control branching between merge and branch node for the same basic block."
              }
            }
            , {
              "type":"inst"
              , "id":21
              , "name":"begin"
              , "file":""
              , "line":""
              , "details":
              {
                "Start-Cycle":"0"
                , "Latency":"1"
                , "Additional Info":"Entrance to a basic block. Control flow comes to this node from one or more branch nodes, unless it's the very first merge node in a kernel. There is no control branching between merge and branch node within the same basic block."
              }
            }
          ]
          , "details":
          {
            "Latency":"24"
          }
        }
      ]
    }
    , {
      "type":"kernel"
      , "id":35
      , "name":"fft1d"
      , "file":""
      , "line":"0"
      , "children":[
        {
          "type":"bb"
          , "id":25
          , "name":"Block2.wii_blk"
          , "file":""
          , "line":"0"
          , "details":
          {
            "Latency":"3"
          }
        }
        , {
          "type":"bb"
          , "id":26
          , "name":"Block3"
          , "file":""
          , "line":"0"
          , "details":
          {
            "Latency":"2"
          }
        }
        , {
          "type":"bb"
          , "id":27
          , "name":"Block4"
          , "file":""
          , "line":"0"
          , "II":1
          , "LoopInfo":""
          , "hasFmaxBottlenecks":"No"
          , "hasSubloops":"No"
          , "isPipelined":"Yes"
          , "children":[
            {
              "type":"inst"
              , "id":29
              , "name":"Channel Read"
              , "file":"2"
              , "line":"60"
              , "details":
              {
                "Width":"256 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"10"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":31
              , "name":"Channel Write"
              , "file":"2"
              , "line":"106"
              , "details":
              {
                "Width":"512 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"131"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":32
              , "name":"loop end"
              , "file":"2"
              , "line":"55"
              , "details":
              {
                "Start-Cycle":"132"
                , "Latency":"1"
                , "Additional Info":"Exit from a basic block. Control flow branches at this node to one or more merge nodes. There is no control branching between merge and branch node for the same basic block."
              }
            }
            , {
              "type":"inst"
              , "id":33
              , "name":"loop"
              , "file":""
              , "line":""
              , "loopTo":32
              , "details":
              {
                "Start-Cycle":"0"
                , "Latency":"1"
                , "Additional Info":"Entrance to a basic block. Control flow comes to this node from one or more branch nodes, unless it's the very first merge node in a kernel. There is no control branching between merge and branch node within the same basic block."
              }
            }
          ]
          , "details":
          {
            "Latency":"133"
          }
        }
        , {
          "type":"bb"
          , "id":28
          , "name":"Block5"
          , "file":""
          , "line":"0"
          , "details":
          {
            "Latency":"2"
          }
        }
      ]
    }
    , {
      "type":"kernel"
      , "id":53
      , "name":"reorder"
      , "file":""
      , "line":"0"
      , "children":[
        {
          "type":"bb"
          , "id":37
          , "name":"Block6"
          , "file":""
          , "line":"0"
          , "children":[
            {
              "type":"inst"
              , "id":38
              , "name":"Channel Read"
              , "file":"4"
              , "line":"50"
              , "details":
              {
                "Width":"256 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"2"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":40
              , "name":"Store"
              , "file":"4"
              , "line":"50"
              , "details":
              {
                "Width":"256 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"3"
                , "Latency":"2"
              }
            }
            , {
              "type":"inst"
              , "id":41
              , "name":"Load"
              , "file":"4"
              , "line":"54"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":42
              , "name":"Load"
              , "file":"4"
              , "line":"55"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":43
              , "name":"Load"
              , "file":"4"
              , "line":"56"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":44
              , "name":"Load"
              , "file":"4"
              , "line":"57"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":45
              , "name":"Load"
              , "file":"4"
              , "line":"58"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":46
              , "name":"Load"
              , "file":"4"
              , "line":"59"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":47
              , "name":"Load"
              , "file":"4"
              , "line":"60"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":48
              , "name":"Load"
              , "file":"4"
              , "line":"61"
              , "details":
              {
                "Width":"32 bits"
                , "Type":"Pipelined"
                , "Stall-free":"Yes"
                , "Start-Cycle":"523"
                , "Latency":"4"
                , "Additional Info":" Part of a stall-free cluster."
              }
            }
            , {
              "type":"inst"
              , "id":49
              , "name":"Channel Write"
              , "file":"4"
              , "line":"62"
              , "details":
              {
                "Width":"256 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"533"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":50
              , "name":"end"
              , "file":"4"
              , "line":"63"
              , "details":
              {
                "Start-Cycle":"534"
                , "Latency":"1"
                , "Additional Info":"Exit from a basic block. Control flow branches at this node to one or more merge nodes. There is no control branching between merge and branch node for the same basic block."
              }
            }
            , {
              "type":"inst"
              , "id":51
              , "name":"begin"
              , "file":""
              , "line":""
              , "details":
              {
                "Start-Cycle":"0"
                , "Latency":"1"
                , "Additional Info":"Entrance to a basic block. Control flow comes to this node from one or more branch nodes, unless it's the very first merge node in a kernel. There is no control branching between merge and branch node within the same basic block."
              }
            }
          ]
          , "details":
          {
            "Latency":"535"
          }
        }
        , {
          "type":"memtype"
          , "id":54
          , "name":"Local Memory"
          , "file":""
          , "line":"0"
          , "children":[
            {
              "type":"memsys"
              , "id":55
              , "name":"buf8"
              , "file":""
              , "line":"0"
              , "replFactor":"24"
              , "banks":1
              , "pumping":1
              , "children":[
                {
                  "type":"bank"
                  , "id":56
                  , "name":"Bank 0"
                  , "file":""
                  , "line":"0"
                }
              ]
            }
          ]
        }
      ]
    }
    , {
      "type":"kernel"
      , "id":67
      , "name":"filter"
      , "file":""
      , "line":"0"
      , "children":[
        {
          "type":"bb"
          , "id":58
          , "name":"Block7.wii_blk"
          , "file":""
          , "line":"0"
          , "details":
          {
            "Latency":"3"
          }
        }
        , {
          "type":"bb"
          , "id":59
          , "name":"Block8"
          , "file":""
          , "line":"0"
          , "details":
          {
            "Latency":"3"
          }
        }
        , {
          "type":"bb"
          , "id":60
          , "name":"Block9"
          , "file":""
          , "line":"0"
          , "II":1
          , "LoopInfo":""
          , "hasFmaxBottlenecks":"No"
          , "hasSubloops":"No"
          , "isPipelined":"Yes"
          , "children":[
            {
              "type":"inst"
              , "id":62
              , "name":"Channel Read"
              , "file":"5"
              , "line":"56"
              , "details":
              {
                "Width":"256 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"10"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":63
              , "name":"Channel Write"
              , "file":"5"
              , "line":"92"
              , "details":
              {
                "Width":"256 bits"
                , "Depth":"8"
                , "Stall-free":"No"
                , "Start-Cycle":"41"
                , "Latency":"1"
              }
            }
            , {
              "type":"inst"
              , "id":64
              , "name":"loop end"
              , "file":"5"
              , "line":"51"
              , "details":
              {
                "Start-Cycle":"42"
                , "Latency":"1"
                , "Additional Info":"Exit from a basic block. Control flow branches at this node to one or more merge nodes. There is no control branching between merge and branch node for the same basic block."
              }
            }
            , {
              "type":"inst"
              , "id":65
              , "name":"loop"
              , "file":""
              , "line":""
              , "loopTo":64
              , "details":
              {
                "Start-Cycle":"0"
                , "Latency":"1"
                , "Additional Info":"Entrance to a basic block. Control flow comes to this node from one or more branch nodes, unless it's the very first merge node in a kernel. There is no control branching between merge and branch node within the same basic block."
              }
            }
          ]
          , "details":
          {
            "Latency":"43"
          }
        }
        , {
          "type":"bb"
          , "id":61
          , "name":"Block10"
          , "file":""
          , "line":"0"
          , "details":
          {
            "Latency":"2"
          }
        }
      ]
    }
    , {
      "type":"memtype"
      , "id":11
      , "name":"Global Memory"
      , "file":""
      , "line":"0"
      , "children":[
        {
          "type":"memsys"
          , "id":12
          , "name":""
          , "file":""
          , "line":"0"
          , "replFactor":"0"
          , "banks":2
          , "pumping":0
          , "children":[
            {
              "type":"bank"
              , "id":13
              , "name":"Bank 0"
              , "file":""
              , "line":"0"
            }
            , {
              "type":"bank"
              , "id":14
              , "name":"Bank 1"
              , "file":""
              , "line":"0"
            }
          ]
        }
      ]
    }
    , {
      "type":"channel"
      , "id":6
      , "name":"DATA_IN"
      , "file":""
      , "line":"0"
      , "details":
      {
        "Width":"256 bits"
        , "Depth":"8"
      }
    }
    , {
      "type":"channel"
      , "id":18
      , "name":"DATA_OUT"
      , "file":""
      , "line":"0"
      , "details":
      {
        "Width":"512 bits"
        , "Depth":"8"
      }
    }
    , {
      "type":"channel"
      , "id":39
      , "name":"FILTER_TO_REORDER"
      , "file":""
      , "line":"0"
      , "details":
      {
        "Width":"256 bits"
        , "Depth":"8"
      }
    }
    , {
      "type":"channel"
      , "id":30
      , "name":"REORDER_TO_FFT"
      , "file":""
      , "line":"0"
      , "details":
      {
        "Width":"256 bits"
        , "Depth":"8"
      }
    }
  ]
  ,
  "links":
  [
    {
      "from":5
      , "to":6
    }
    ,
    {
      "from":4
      , "to":5
    }
    ,
    {
      "from":8
      , "to":4
    }
    ,
    {
      "from":5
      , "to":7
    }
    ,
    {
      "from":13
      , "to":4
    }
    ,
    {
      "from":14
      , "to":4
    }
    ,
    {
      "from":18
      , "to":17
    }
    ,
    {
      "from":21
      , "to":17
    }
    ,
    {
      "from":17
      , "to":19
    }
    ,
    {
      "from":19
      , "to":20
    }
    ,
    {
      "from":19
      , "to":13
    }
    ,
    {
      "from":19
      , "to":14
    }
    ,
    {
      "from":30
      , "to":29
    }
    ,
    {
      "from":31
      , "to":18
    }
    ,
    {
      "from":33
      , "to":29
    }
    ,
    {
      "from":29
      , "to":31
    }
    ,
    {
      "from":32
      , "to":33
    }
    ,
    {
      "from":26
      , "to":33
    }
    ,
    {
      "from":32
      , "to":28
    }
    ,
    {
      "from":31
      , "to":32
    }
    ,
    {
      "from":25
      , "to":26
    }
    ,
    {
      "from":39
      , "to":38
    }
    ,
    {
      "from":49
      , "to":30
    }
    ,
    {
      "from":56
      , "to":41
    }
    ,
    {
      "from":56
      , "to":42
    }
    ,
    {
      "from":56
      , "to":43
    }
    ,
    {
      "from":56
      , "to":44
    }
    ,
    {
      "from":56
      , "to":45
    }
    ,
    {
      "from":56
      , "to":46
    }
    ,
    {
      "from":56
      , "to":47
    }
    ,
    {
      "from":56
      , "to":48
    }
    ,
    {
      "from":40
      , "to":56
    }
    ,
    {
      "from":40
      , "to":41
    }
    ,
    {
      "from":38
      , "to":40
    }
    ,
    {
      "from":41
      , "to":49
    }
    ,
    {
      "from":42
      , "to":49
    }
    ,
    {
      "from":43
      , "to":49
    }
    ,
    {
      "from":44
      , "to":49
    }
    ,
    {
      "from":45
      , "to":49
    }
    ,
    {
      "from":46
      , "to":49
    }
    ,
    {
      "from":47
      , "to":49
    }
    ,
    {
      "from":48
      , "to":49
    }
    ,
    {
      "from":40
      , "to":49
    }
    ,
    {
      "from":40
      , "to":47
    }
    ,
    {
      "from":40
      , "to":48
    }
    ,
    {
      "from":40
      , "to":46
    }
    ,
    {
      "from":40
      , "to":43
    }
    ,
    {
      "from":40
      , "to":42
    }
    ,
    {
      "from":40
      , "to":44
    }
    ,
    {
      "from":40
      , "to":45
    }
    ,
    {
      "from":51
      , "to":38
    }
    ,
    {
      "from":49
      , "to":50
    }
    ,
    {
      "from":6
      , "to":62
    }
    ,
    {
      "from":63
      , "to":39
    }
    ,
    {
      "from":65
      , "to":62
    }
    ,
    {
      "from":62
      , "to":63
    }
    ,
    {
      "from":63
      , "to":64
    }
    ,
    {
      "from":64
      , "to":61
    }
    ,
    {
      "from":58
      , "to":59
    }
    ,
    {
      "from":64
      , "to":65
    }
    ,
    {
      "from":59
      , "to":65
    }
  ]
  , "fileIndexMap":
  {
    "/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl":"1"
    , "/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl":"2"
    , "/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl":"3"
    , "/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl":"4"
    , "/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl":"5"
  }
}
;var areaJSON={
  "columns":["ALUTs", "FFs", "RAMs", "DSPs"]
  , "debug_enabled":1
  , "total_percent":
  [54.1845, 31.8822, 24.8542, 41.4453, 76.5625]
  , "total":
  [149668, 233351, 1061, 196]
  , "name":"Kernel System"
  , "max_resources":
  [469440, 938880, 2560, 256]
  , "partitions":
  [
  ]
  , "resources":
  [
    {
      "name":"Board interface"
      , "data":
      [39076, 51471, 283, 0]
      , "details":
      [
        "Platform interface logic."
      ]
    }
    , {
      "name":"Global interconnect"
      , "data":
      [5034, 9568, 52, 0]
      , "details":
      [
        "Global interconnect for 1 global load and 1 global store."
      ]
    }
    , {
      "name":"channelizer.cl:47 (DATA_IN)"
      , "data":
      [58, 1072, 7, 0]
      , "debug":
      [
        [
          {
            "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
            , "line":47
          }
        ]
      ]
      , "details":
      [
        "Channel is implemented 256 bits wide by 16 deep. Requested depth was 8.\nChannel depth was changed for the following reasons:\n- instruction scheduling requirements\n- nature of underlying FIFO implementation"
      ]
    }
    , {
      "name":"channelizer.cl:48 (FILTER_TO_REORDER)"
      , "data":
      [58, 1072, 7, 0]
      , "debug":
      [
        [
          {
            "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
            , "line":48
          }
        ]
      ]
      , "details":
      [
        "Channel is implemented 256 bits wide by 16 deep. Requested depth was 8.\nChannel depth was changed for the following reasons:\n- instruction scheduling requirements\n- nature of underlying FIFO implementation"
      ]
    }
    , {
      "name":"channelizer.cl:49 (REORDER_TO_FFT)"
      , "data":
      [58, 1072, 7, 0]
      , "debug":
      [
        [
          {
            "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
            , "line":49
          }
        ]
      ]
      , "details":
      [
        "Channel is implemented 256 bits wide by 16 deep. Requested depth was 8.\nChannel depth was changed for the following reasons:\n- instruction scheduling requirements\n- nature of underlying FIFO implementation"
      ]
    }
    , {
      "name":"channelizer.cl:50 (DATA_OUT)"
      , "data":
      [61, 2099, 13, 0]
      , "debug":
      [
        [
          {
            "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
            , "line":50
          }
        ]
      ]
      , "details":
      [
        "Channel is implemented 512 bits wide by 16 deep. Requested depth was 8.\nChannel depth was changed for the following reasons:\n- instruction scheduling requirements\n- nature of underlying FIFO implementation"
      ]
    }
  ]
  , "functions":
  [
    {
      "name":"data_in"
      , "compute_units":1
      , "details":
      [
        "Number of compute units: 1"
      ]
      , "resources":
      [
        {
          "name":"Function overhead"
          , "data":
          [1570, 1685, 0, 0]
          , "details":
          [
            "Kernel dispatch logic."
          ]
        }
      ]
      , "basicblocks":
      [
        {
          "name":"Block0"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [277, 450, 0, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [12, 12, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"channelizer.cl:74"
                    , "data":
                    [9, 182, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                          , "line":74
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"channelizer.cl:75"
                    , "data":
                    [256, 256, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                          , "line":75
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
          ]
          , "computation":
          [
            {
              "name":"channelizer.cl:74"
              , "data":
              [435, 1961, 13, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                    , "line":74
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [429, 1961, 13, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Pointer Computation"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":1
                }
              ]
            }
          ]
        }
      ]
    }
    , {
      "name":"data_out"
      , "compute_units":1
      , "details":
      [
        "Number of compute units: 1"
      ]
      , "resources":
      [
        {
          "name":"Function overhead"
          , "data":
          [1570, 1685, 0, 0]
          , "details":
          [
            "Kernel dispatch logic."
          ]
        }
      ]
      , "basicblocks":
      [
        {
          "name":"Block1"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [2246, 4844, 1, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [9, 9, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"channelizer.cl:80"
                    , "data":
                    [712, 1506, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                          , "line":80
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"channelizer.cl:81"
                    , "data":
                    [1474, 3096, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                          , "line":81
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"channelizer.cl:82"
                    , "data":
                    [51, 233, 1, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                          , "line":82
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
            , {
              "name":"Cluster logic"
              , "data":
              [218, 502, 2, 0]
              , "details":
              [
                "Logic required to efficiently support sets of operations that do not stall. This area cannot be affected directly."
              ]
            }
          ]
          , "computation":
          [
            {
              "name":"channelizer.cl:80"
              , "data":
              [224, 768, 0, 16]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                    , "line":80
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [64, 0, 0, 0]
                  }
                  , "count":48
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [32, 0, 0, 0]
                  }
                  , "count":336
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 768, 0, 16]
                  }
                  , "count":16
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [128, 0, 0, 0]
                  }
                  , "count":128
                }
              ]
            }
            , {
              "name":"channelizer.cl:81"
              , "data":
              [824, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                    , "line":81
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [360, 0, 0, 0]
                  }
                  , "count":40
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":456
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [240, 0, 0, 0]
                  }
                  , "count":192
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":128
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":16
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [104, 0, 0, 0]
                  }
                  , "count":80
                }
              ]
            }
            , {
              "name":"channelizer.cl:82"
              , "data":
              [352, 2436, 16, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/channelizer.cl"
                    , "line":82
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Pointer Computation"
                    , "data":
                    [4, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Store"
                    , "data":
                    [348, 2436, 16, 0]
                  }
                  , "count":1
                }
              ]
            }
          ]
        }
      ]
    }
    , {
      "name":"fft1d"
      , "compute_units":1
      , "details":
      [
        "Number of compute units: 1"
      ]
      , "resources":
      [
        {
          "name":"Function overhead"
          , "data":
          [1570, 1685, 0, 0]
          , "details":
          [
            "Kernel dispatch logic."
          ]
        }
        , {
          "name":"Private Variable: \n - 'fft_delay_elements' (fft1d.cl:44)"
          , "data":
          [1795, 4652, 81, 0]
          , "debug":
          [
            [
              {
                "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                , "line":44
              }
            ]
          ]
          , "details":
          [
            "Implemented as a shift register with 34 or fewer tap points. This is a very efficient storage type.\nImplemented using registers of the following sizes:\n- 1 register of width 9 and depth 1\n- 8 registers of width 64 and depth 4\n- 6 registers of width 64 and depth 8\n- 2 registers of width 64 and depth 9\n- 6 registers of width 64 and depth 16\n- 2 registers of width 64 and depth 17\n- 8 registers of width 64 and depth 32\n- 6 registers of width 64 and depth 64\n- 2 registers of width 64 and depth 65\n- 8 registers of width 64 and depth 128\n- 8 registers of width 64 and depth 256"
          ]
        }
        , {
          "name":"Private Variable: \n - 'i' (fft1d.cl:55)"
          , "data":
          [8, 69, 0, 0]
          , "debug":
          [
            [
              {
                "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                , "line":55
              }
            ]
          ]
          , "details":
          [
            "Implemented using registers of the following size:\n- 1 register of width 32 and depth 1"
          ]
        }
      ]
      , "basicblocks":
      [
        {
          "name":"Block3"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [0, 55, 0, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [0, 23, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:338"
                    , "data":
                    [0, 32, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":338
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
          ]
          , "computation":
          [
          ]
        }
        , {
          "name":"Block4"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [44480.4, 93238.1, 24, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [2, 2, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"No Source Line"
                    , "data":
                    [222.537, 309.086, 7.74527, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:106"
                    , "data":
                    [33, 26, 1, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":106
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:44"
                    , "data":
                    [6.96091, 15.0696, 0.207799, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":44
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:55"
                    , "data":
                    [33, 30.4333, 1, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":55
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:61"
                    , "data":
                    [108.129, 216.424, 0.293252, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":61
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:63"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":63
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:64"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":64
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:65"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":65
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:66"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":66
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:67"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":67
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:68"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":68
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:69"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":69
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:70"
                    , "data":
                    [146.5, 307.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":280
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":70
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:63"
                    , "data":
                    [111.5, 237, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":285
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":63
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:64"
                    , "data":
                    [111.5, 237, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":285
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":64
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:67"
                    , "data":
                    [111.5, 237, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":285
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":67
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:68"
                    , "data":
                    [111.5, 237, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":285
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":68
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286"
                    , "data":
                    [4.8, 9.6, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:181"
                    , "data":
                    [40.5, 86.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:182"
                    , "data":
                    [40.5, 86.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:214"
                    , "data":
                    [23.25, 46.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":214
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:216"
                    , "data":
                    [23.25, 46.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":216
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:250 > fft_8.cl:181"
                    , "data":
                    [309.5, 665, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:250 > fft_8.cl:182"
                    , "data":
                    [308.5, 664, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:251 > fft_8.cl:181"
                    , "data":
                    [304.5, 648, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:251 > fft_8.cl:182"
                    , "data":
                    [303.5, 647, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:181"
                    , "data":
                    [40.5, 86.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:182"
                    , "data":
                    [40.5, 86.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:214"
                    , "data":
                    [23.25, 46.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":214
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:216"
                    , "data":
                    [23.25, 46.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":216
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:253 > fft_8.cl:181"
                    , "data":
                    [309.5, 665, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":253
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:253 > fft_8.cl:182"
                    , "data":
                    [308.5, 664, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":253
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:254 > fft_8.cl:181"
                    , "data":
                    [304.5, 648, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":254
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:254 > fft_8.cl:182"
                    , "data":
                    [303.5, 647, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":286
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":254
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:301"
                    , "data":
                    [98.6444, 159.956, 1.94444, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":301
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:63"
                    , "data":
                    [2714.29, 5603.06, 0.57996, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":63
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:64"
                    , "data":
                    [2627.94, 5429.4, 0.519765, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":64
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:65"
                    , "data":
                    [2415.5, 5123.73, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":65
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:66"
                    , "data":
                    [2418, 5122.41, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":66
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:67"
                    , "data":
                    [2546.3, 5362.62, 0.679313, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":67
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:68"
                    , "data":
                    [2522.19, 5315.67, 0.529909, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":68
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:69"
                    , "data":
                    [2446.11, 5162.34, 0.108047, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":69
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:70"
                    , "data":
                    [2447.59, 5163.1, 0.0599696, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":70
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:181"
                    , "data":
                    [1334.53, 2862.39, 0.0399798, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:182"
                    , "data":
                    [1267.92, 2717.85, 0.0597447, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:214"
                    , "data":
                    [20.6667, 41.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":214
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:216"
                    , "data":
                    [20.6667, 41.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":216
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:181"
                    , "data":
                    [1321.8, 2833.93, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:182"
                    , "data":
                    [1250.73, 2682.79, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:214"
                    , "data":
                    [20.6667, 41.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":214
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:216"
                    , "data":
                    [20.6667, 41.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":216
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:181"
                    , "data":
                    [1339.73, 2866.4, 0.0666667, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:182"
                    , "data":
                    [1271.17, 2719.58, 0.0666667, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:214"
                    , "data":
                    [20.6667, 41.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":214
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:216"
                    , "data":
                    [20.6667, 41.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":216
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:252 > fft_8.cl:181"
                    , "data":
                    [1385.93, 2904.08, 0.379858, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:252 > fft_8.cl:182"
                    , "data":
                    [1426.21, 2968.66, 0.699275, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:253 > fft_8.cl:181"
                    , "data":
                    [1382.2, 2901.18, 0.2, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":253
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:253 > fft_8.cl:182"
                    , "data":
                    [1420.93, 2960.7, 0.119939, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":253
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:254 > fft_8.cl:181"
                    , "data":
                    [1380.33, 2900.7, 0.0666667, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":254
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:254 > fft_8.cl:182"
                    , "data":
                    [1417.67, 2961.46, 0.0666667, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":254
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:315"
                    , "data":
                    [193.556, 239.444, 7.05556, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":315
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:150 > fft_8.cl:137"
                    , "data":
                    [0.179487, 0.557447, 0.0128205, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":322
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":150
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":137
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:151 > fft_8.cl:137"
                    , "data":
                    [0.449123, 1.05466, 0.019765, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":322
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":151
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":137
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:152 > fft_8.cl:137"
                    , "data":
                    [0, 1.23452, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":322
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":152
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":137
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:153 > fft_8.cl:137"
                    , "data":
                    [2.5, 3.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":322
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":153
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":137
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:155"
                    , "data":
                    [461.171, 949.158, 0.270861, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":322
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":155
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:63"
                    , "data":
                    [307, 641, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":63
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:64"
                    , "data":
                    [307, 641, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":64
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:65"
                    , "data":
                    [320, 668, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":65
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:66"
                    , "data":
                    [320, 668, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":66
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:67"
                    , "data":
                    [320, 668, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":67
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:68"
                    , "data":
                    [320, 668, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":68
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:69"
                    , "data":
                    [313.5, 654.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":69
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:70"
                    , "data":
                    [313.5, 654.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":330
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":70
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:338"
                    , "data":
                    [6.96091, 15.4363, 0.207799, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":338
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
            , {
              "name":"Feedback"
              , "data":
              [944, 7532, 0, 0]
              , "details":
              [
                "Resources for loop-carried dependencies. To reduce this area:\n- reduce number and size of loop-carried variables"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"fft1d.cl:55"
                    , "data":
                    [34, 176, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":55
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:61"
                    , "data":
                    [96, 830.667, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":61
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:63"
                    , "data":
                    [5.33333, 174, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":63
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:67"
                    , "data":
                    [25.3333, 644.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":67
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:68"
                    , "data":
                    [20, 470.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":68
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:69"
                    , "data":
                    [4, 258.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":69
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:70"
                    , "data":
                    [4, 258.5, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":303
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":70
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:181"
                    , "data":
                    [2.66667, 44.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:182"
                    , "data":
                    [13.3333, 65.6667, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":249
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:181"
                    , "data":
                    [5.33333, 131.333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:182"
                    , "data":
                    [37.3333, 195.333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":250
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:181"
                    , "data":
                    [5.33333, 88.6667, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:182"
                    , "data":
                    [26.6667, 131.333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":251
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:252 > fft_8.cl:181"
                    , "data":
                    [2.66667, 44.3333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:252 > fft_8.cl:182"
                    , "data":
                    [111.333, 570.667, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":252
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:253 > fft_8.cl:181"
                    , "data":
                    [149.333, 909.333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":253
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:253 > fft_8.cl:182"
                    , "data":
                    [37.3333, 195.333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":253
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:254 > fft_8.cl:181"
                    , "data":
                    [31.3333, 204.667, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":254
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":181
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:254 > fft_8.cl:182"
                    , "data":
                    [156.667, 828.333, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":306
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":254
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":182
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:155"
                    , "data":
                    [176, 1310, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                          , "line":83
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":322
                        }
                        , {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                          , "line":155
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
            , {
              "name":"Cluster logic"
              , "data":
              [461, 1058, 5, 0]
              , "details":
              [
                "Logic required to efficiently support sets of operations that do not stall. This area cannot be affected directly."
              ]
            }
          ]
          , "computation":
          [
            {
              "name":"No Source Line"
              , "data":
              [324, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":""
                    , "line":0
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [324, 0, 0, 0]
                  }
                  , "count":162
                }
              ]
            }
            , {
              "name":"fft1d.cl:59"
              , "data":
              [11, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":59
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:63"
              , "data":
              [88, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":63
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":33
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":17
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [23, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:64"
              , "data":
              [88, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":64
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":33
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":17
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [23, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:65"
              , "data":
              [91.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":65
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":37
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [26.5, 0, 0, 0]
                  }
                  , "count":20
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:66"
              , "data":
              [91.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":66
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":37
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [26.5, 0, 0, 0]
                  }
                  , "count":20
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:67"
              , "data":
              [88, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":67
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":33
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":17
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [23, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:68"
              , "data":
              [88, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":68
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":33
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":17
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [23, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:69"
              , "data":
              [91.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":69
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":37
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [26.5, 0, 0, 0]
                  }
                  , "count":20
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:280 > \nfft_8.cl:70"
              , "data":
              [91.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":280
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":70
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":37
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [16, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [26.5, 0, 0, 0]
                  }
                  , "count":20
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [18.5, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:63"
              , "data":
              [78, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":285
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":63
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6.5, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":15
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":12
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:64"
              , "data":
              [78, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":285
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":64
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6.5, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":15
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":12
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:67"
              , "data":
              [78, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":285
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":67
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6.5, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":15
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":12
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:285 > \nfft_8.cl:68"
              , "data":
              [78, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":285
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":68
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [15.5, 0, 0, 0]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [6.5, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [9, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":15
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [17, 0, 0, 0]
                  }
                  , "count":12
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":6
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:181"
              , "data":
              [19, 51, 0, 1]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [4, 0, 0, 0]
                  }
                  , "count":3
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [2, 0, 0, 0]
                  }
                  , "count":23
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 51, 0, 1]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":11
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:182"
              , "data":
              [19, 51, 0, 1]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [4, 0, 0, 0]
                  }
                  , "count":3
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [2, 0, 0, 0]
                  }
                  , "count":23
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 51, 0, 1]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":11
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:214"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:249 > fft_8.cl:216"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:250 > fft_8.cl:181"
              , "data":
              [155, 99, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":107
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":41
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 99, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [43, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [27, 0, 0, 0]
                  }
                  , "count":22
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:250 > fft_8.cl:182"
              , "data":
              [137, 99, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":98
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":38
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 99, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [25, 0, 0, 0]
                  }
                  , "count":35
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [27, 0, 0, 0]
                  }
                  , "count":21
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:250 > fft_8.cl:214"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:250 > fft_8.cl:216"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:251 > fft_8.cl:181"
              , "data":
              [141, 102, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":106
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":40
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 102, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":41
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [19, 0, 0, 0]
                  }
                  , "count":22
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:251 > fft_8.cl:182"
              , "data":
              [129, 102, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":99
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":38
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 102, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [25, 0, 0, 0]
                  }
                  , "count":35
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [19, 0, 0, 0]
                  }
                  , "count":21
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:251 > fft_8.cl:214"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:251 > fft_8.cl:216"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:181"
              , "data":
              [19, 51, 0, 1]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":252
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [4, 0, 0, 0]
                  }
                  , "count":3
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [2, 0, 0, 0]
                  }
                  , "count":23
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 51, 0, 1]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":11
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:182"
              , "data":
              [19, 51, 0, 1]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":252
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [4, 0, 0, 0]
                  }
                  , "count":3
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [2, 0, 0, 0]
                  }
                  , "count":23
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 51, 0, 1]
                  }
                  , "count":1
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":11
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:214"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":252
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:252 > fft_8.cl:216"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":252
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:253 > fft_8.cl:181"
              , "data":
              [155, 99, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":253
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":107
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":41
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 99, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [43, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [27, 0, 0, 0]
                  }
                  , "count":22
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:253 > fft_8.cl:182"
              , "data":
              [137, 99, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":253
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":98
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":38
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 99, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [25, 0, 0, 0]
                  }
                  , "count":35
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [27, 0, 0, 0]
                  }
                  , "count":21
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:253 > fft_8.cl:214"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":253
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:253 > fft_8.cl:216"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":253
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:254 > fft_8.cl:181"
              , "data":
              [141, 102, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":254
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":106
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":40
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 102, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":41
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [19, 0, 0, 0]
                  }
                  , "count":22
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:254 > fft_8.cl:182"
              , "data":
              [129, 102, 0, 2]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":254
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [38, 0, 0, 0]
                  }
                  , "count":11
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [11, 0, 0, 0]
                  }
                  , "count":99
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [30, 0, 0, 0]
                  }
                  , "count":38
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 102, 0, 2]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [25, 0, 0, 0]
                  }
                  , "count":35
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [6, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [19, 0, 0, 0]
                  }
                  , "count":21
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:254 > fft_8.cl:214"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":254
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:286 > \nfft_8.cl:254 > fft_8.cl:216"
              , "data":
              [0, 0, 2, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":286
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":254
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 2, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:63"
              , "data":
              [1507.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":63
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [263.5, 0, 0, 0]
                  }
                  , "count":17
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [110, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [153, 0, 0, 0]
                  }
                  , "count":575
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [275, 0, 0, 0]
                  }
                  , "count":279
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [397, 0, 0, 0]
                  }
                  , "count":276
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [309, 0, 0, 0]
                  }
                  , "count":102
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:64"
              , "data":
              [1465.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":64
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [263.5, 0, 0, 0]
                  }
                  , "count":17
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [110, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [153, 0, 0, 0]
                  }
                  , "count":551
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [275, 0, 0, 0]
                  }
                  , "count":273
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [355, 0, 0, 0]
                  }
                  , "count":264
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [309, 0, 0, 0]
                  }
                  , "count":103
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:65"
              , "data":
              [1550.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":65
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [279, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [116.5, 0, 0, 0]
                  }
                  , "count":36
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [162, 0, 0, 0]
                  }
                  , "count":585
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [292, 0, 0, 0]
                  }
                  , "count":289
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [379, 0, 0, 0]
                  }
                  , "count":271
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [322, 0, 0, 0]
                  }
                  , "count":109
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:66"
              , "data":
              [1550.5, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":66
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [279, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [116.5, 0, 0, 0]
                  }
                  , "count":36
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [162, 0, 0, 0]
                  }
                  , "count":585
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [292, 0, 0, 0]
                  }
                  , "count":289
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [379, 0, 0, 0]
                  }
                  , "count":271
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [322, 0, 0, 0]
                  }
                  , "count":109
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:67"
              , "data":
              [1608, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":67
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [279, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [117, 0, 0, 0]
                  }
                  , "count":42
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [162, 0, 0, 0]
                  }
                  , "count":634
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [292, 0, 0, 0]
                  }
                  , "count":313
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [425, 0, 0, 0]
                  }
                  , "count":290
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [333, 0, 0, 0]
                  }
                  , "count":109
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:68"
              , "data":
              [1608, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":68
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [279, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [117, 0, 0, 0]
                  }
                  , "count":36
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [162, 0, 0, 0]
                  }
                  , "count":634
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [292, 0, 0, 0]
                  }
                  , "count":306
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [425, 0, 0, 0]
                  }
                  , "count":290
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [333, 0, 0, 0]
                  }
                  , "count":109
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:69"
              , "data":
              [1587, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":69
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [279, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [117, 0, 0, 0]
                  }
                  , "count":36
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [162, 0, 0, 0]
                  }
                  , "count":616
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [292, 0, 0, 0]
                  }
                  , "count":300
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [404, 0, 0, 0]
                  }
                  , "count":276
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [333, 0, 0, 0]
                  }
                  , "count":109
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:303 > \nfft_8.cl:70"
              , "data":
              [1587, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":303
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":70
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [279, 0, 0, 0]
                  }
                  , "count":18
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [117, 0, 0, 0]
                  }
                  , "count":36
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [162, 0, 0, 0]
                  }
                  , "count":616
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [292, 0, 0, 0]
                  }
                  , "count":300
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [404, 0, 0, 0]
                  }
                  , "count":276
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [333, 0, 0, 0]
                  }
                  , "count":109
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:181"
              , "data":
              [585, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [44, 0, 0, 0]
                  }
                  , "count":436
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":163
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [169, 0, 0, 0]
                  }
                  , "count":170
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":88
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:182"
              , "data":
              [537, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [44, 0, 0, 0]
                  }
                  , "count":408
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [121, 0, 0, 0]
                  }
                  , "count":146
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":9
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":85
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:214"
              , "data":
              [0, 0, 8, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 8, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":4
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:249 > fft_8.cl:216"
              , "data":
              [0, 0, 8, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":249
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 8, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":4
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:181"
              , "data":
              [585, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [44, 0, 0, 0]
                  }
                  , "count":436
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":163
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [169, 0, 0, 0]
                  }
                  , "count":170
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":88
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:182"
              , "data":
              [537, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [44, 0, 0, 0]
                  }
                  , "count":408
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [121, 0, 0, 0]
                  }
                  , "count":146
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":9
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":85
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:214"
              , "data":
              [0, 0, 8, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 8, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":4
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:250 > fft_8.cl:216"
              , "data":
              [0, 0, 8, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":250
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 8, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":4
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:181"
              , "data":
              [585, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [44, 0, 0, 0]
                  }
                  , "count":436
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":163
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [169, 0, 0, 0]
                  }
                  , "count":170
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":88
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:182"
              , "data":
              [537, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [44, 0, 0, 0]
                  }
                  , "count":408
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [121, 0, 0, 0]
                  }
                  , "count":146
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":9
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":85
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:214"
              , "data":
              [0, 0, 8, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":214
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 8, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":4
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:251 > fft_8.cl:216"
              , "data":
              [0, 0, 8, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":251
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":216
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 8, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":4
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:252 > fft_8.cl:181"
              , "data":
              [516, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":252
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [41, 0, 0, 0]
                  }
                  , "count":403
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [103, 0, 0, 0]
                  }
                  , "count":140
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":88
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:252 > fft_8.cl:182"
              , "data":
              [516, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":252
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [41, 0, 0, 0]
                  }
                  , "count":399
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [103, 0, 0, 0]
                  }
                  , "count":140
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":9
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":85
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:253 > fft_8.cl:181"
              , "data":
              [516, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":253
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [41, 0, 0, 0]
                  }
                  , "count":403
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [103, 0, 0, 0]
                  }
                  , "count":140
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":88
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:253 > fft_8.cl:182"
              , "data":
              [516, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":253
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [41, 0, 0, 0]
                  }
                  , "count":399
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [103, 0, 0, 0]
                  }
                  , "count":140
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":9
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":85
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:254 > fft_8.cl:181"
              , "data":
              [516, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":254
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":181
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [41, 0, 0, 0]
                  }
                  , "count":403
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [103, 0, 0, 0]
                  }
                  , "count":140
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":88
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:306 > \nfft_8.cl:254 > fft_8.cl:182"
              , "data":
              [516, 408, 0, 8]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":306
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":254
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":182
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [152, 0, 0, 0]
                  }
                  , "count":44
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [41, 0, 0, 0]
                  }
                  , "count":399
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [120, 0, 0, 0]
                  }
                  , "count":155
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 408, 0, 8]
                  }
                  , "count":8
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [103, 0, 0, 0]
                  }
                  , "count":140
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [24, 0, 0, 0]
                  }
                  , "count":9
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [76, 0, 0, 0]
                  }
                  , "count":85
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:322 > \nfft_8.cl:155"
              , "data":
              [112, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":322
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":155
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [112, 0, 0, 0]
                  }
                  , "count":32
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:63"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":63
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:64"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":64
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:65"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":65
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:66"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":66
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:67"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":67
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:68"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":68
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:69"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":69
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
            , {
              "name":"fft1d.cl:83 > fft_8.cl:330 > \nfft_8.cl:70"
              , "data":
              [181, 0, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft1d.cl"
                    , "line":83
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":330
                  }
                  , {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/fft_8.cl"
                    , "line":70
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'llvm.ctpop.i32' Function Call"
                    , "data":
                    [31, 0, 0, 0]
                  }
                  , "count":2
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [13, 0, 0, 0]
                  }
                  , "count":4
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [18, 0, 0, 0]
                  }
                  , "count":72
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [34, 0, 0, 0]
                  }
                  , "count":34
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [48, 0, 0, 0]
                  }
                  , "count":30
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [37, 0, 0, 0]
                  }
                  , "count":12
                }
              ]
            }
          ]
        }
      ]
    }
    , {
      "name":"filter"
      , "compute_units":1
      , "details":
      [
        "Number of compute units: 1"
      ]
      , "resources":
      [
        {
          "name":"Function overhead"
          , "data":
          [1570, 1685, 0, 0]
          , "details":
          [
            "Kernel dispatch logic."
          ]
        }
        , {
          "name":"Coalesced Private Variables: \n - 'samples' (filter.cl:48)\n - 'i' (filter.cl:51)"
          , "data":
          [7, 88.375, 0, 0]
          , "debug":
          [
            [
              {
                "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                , "line":48
              }
            ]
            , [
              {
                "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                , "line":51
              }
            ]
          ]
          , "details":
          [
            "Implemented using registers of the following sizes:\n- 1 register of width 10 and depth 1\n- 1 register of width 32 and depth 1"
          ]
        }
        , {
          "name":"Private Variable: \n - 'samples' (filter.cl:48)"
          , "data":
          [331, 289, 85, 0]
          , "debug":
          [
            [
              {
                "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                , "line":48
              }
            ]
          ]
          , "details":
          [
            "Implemented as a shift register with 7 or fewer tap points. This is a very efficient storage type.\nImplemented using registers of the following sizes:\n- 8 registers of width 32 and depth 512\n- 48 registers of width 32 and depth 513"
          ]
        }
      ]
      , "basicblocks":
      [
        {
          "name":"Block8"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [8, 39, 0, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [0, 1, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:59"
                    , "data":
                    [8, 38, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":59
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
          ]
          , "computation":
          [
          ]
        }
        , {
          "name":"Block9"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [12929, 27684.1, 2, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [2, 2, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"No Source Line"
                    , "data":
                    [5019.2, 10863.3, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:48"
                    , "data":
                    [42.6667, 87.4318, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":48
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:51"
                    , "data":
                    [33, 31.2524, 1, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":51
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:56"
                    , "data":
                    [128, 256, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":56
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:59"
                    , "data":
                    [42.6667, 87.7413, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":59
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:63"
                    , "data":
                    [42.6667, 87.4318, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":63
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:87"
                    , "data":
                    [7585.8, 16242.8, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":87
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"filter.cl:92"
                    , "data":
                    [33, 26, 1, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":92
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
            , {
              "name":"Feedback"
              , "data":
              [35, 188.625, 0, 0]
              , "details":
              [
                "Resources for loop-carried dependencies. To reduce this area:\n- reduce number and size of loop-carried variables"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"filter.cl:51"
                    , "data":
                    [35, 188.625, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                          , "line":51
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
            , {
              "name":"Cluster logic"
              , "data":
              [477, 1079, 5, 0]
              , "details":
              [
                "Logic required to efficiently support sets of operations that do not stall. This area cannot be affected directly."
              ]
            }
          ]
          , "computation":
          [
            {
              "name":"No Source Line"
              , "data":
              [2160, 0, 128, 0]
              , "debug":
              [
                [
                  {
                    "filename":""
                    , "line":0
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'__acl__optimized_clz_27' Function Call"
                    , "data":
                    [816, 0, 0, 0]
                  }
                  , "count":48
                }
                , {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [576, 0, 0, 0]
                  }
                  , "count":120
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [384, 0, 0, 0]
                  }
                  , "count":288
                }
                , {
                  "info":
                  {
                    "name":"On-chip Read-Only Memory Lookup"
                    , "data":
                    [0, 0, 128, 0]
                    , "details":
                    [
                      "Read from 16384 bit ROM. A copy of the ROM is created for each access."
                    ]
                  }
                  , "count":64
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [384, 0, 0, 0]
                  }
                  , "count":96
                }
              ]
            }
            , {
              "name":"filter.cl:87"
              , "data":
              [5024, 3072, 0, 64]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/filter.cl"
                    , "line":87
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Add"
                    , "data":
                    [1216, 0, 0, 0]
                  }
                  , "count":304
                }
                , {
                  "info":
                  {
                    "name":"And"
                    , "data":
                    [176, 0, 0, 0]
                  }
                  , "count":2536
                }
                , {
                  "info":
                  {
                    "name":"Integer Compare"
                    , "data":
                    [960, 0, 0, 0]
                  }
                  , "count":944
                }
                , {
                  "info":
                  {
                    "name":"Mul"
                    , "data":
                    [0, 3072, 0, 64]
                  }
                  , "count":64
                }
                , {
                  "info":
                  {
                    "name":"Or"
                    , "data":
                    [1408, 0, 0, 0]
                  }
                  , "count":1144
                }
                , {
                  "info":
                  {
                    "name":"Sub"
                    , "data":
                    [144, 0, 0, 0]
                  }
                  , "count":40
                }
                , {
                  "info":
                  {
                    "name":"Xor"
                    , "data":
                    [1120, 0, 0, 0]
                  }
                  , "count":504
                }
              ]
            }
          ]
        }
      ]
    }
    , {
      "name":"reorder"
      , "compute_units":1
      , "details":
      [
        "Number of compute units: 1"
      ]
      , "resources":
      [
        {
          "name":"Function overhead"
          , "data":
          [1623, 1789, 0, 0]
          , "details":
          [
            "Kernel dispatch logic."
          ]
        }
        , {
          "name":"reorder.cl:45 (buf8)"
          , "data":
          [0, 0, 256, 0]
          , "debug":
          [
            [
              {
                "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                , "line":45
              }
            ]
          ]
          , "details":
          [
            "Local memory: Good but replicated.\nRequested size 16384 bytes (rounded up to nearest power of 2), implemented size 393216 bytes, replicated 24 times total, stall-free, 8 reads and 1 write. Additional information:\n- Replicated 3 times to efficiently support multiple simultaneous workgroups. This replication resulted in 4 times increase in actual block RAM usage. Reducing the number of barriers or increasing max_work_group_size may help reduce this replication factor.\n- Replicated 8 times to efficiently support multiple accesses. To reduce this replication factor, reduce number of read and write accesses."
          ]
        }
      ]
      , "basicblocks":
      [
        {
          "name":"Block6"
          , "resources":
          [
            {
              "name":"State"
              , "data":
              [679.2, 1598.4, 0, 0]
              , "details":
              [
                "Resources for live values and control logic. To reduce this area:\n- reduce size of local variables\n- reduce scope of local variables, localizing them as much as possible\n- reduce number of nested loops"
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Control flow logic"
                    , "data":
                    [32, 32, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"No Source Line"
                    , "data":
                    [48, 96, 0, 0]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:50"
                    , "data":
                    [278.2, 828.4, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":50
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:54"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":54
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:55"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":55
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:56"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":56
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:57"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":57
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:58"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":58
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:59"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":59
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:60"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":60
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:61"
                    , "data":
                    [24.125, 48.25, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":61
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
                , {
                  "info":
                  {
                    "name":"reorder.cl:62"
                    , "data":
                    [128, 256, 0, 0]
                    , "debug":
                    [
                      [
                        {
                          "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                          , "line":62
                        }
                      ]
                    ]
                  }
                  , "count":0
                }
              ]
            }
            , {
              "name":"Cluster logic"
              , "data":
              [238.8, 799.6, 2, 0]
              , "details":
              [
                "Logic required to efficiently support sets of operations that do not stall. This area cannot be affected directly."
              ]
            }
          ]
          , "computation":
          [
            {
              "name":"reorder.cl:50"
              , "data":
              [34, 24, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":50
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Store"
                    , "data":
                    [34, 24, 0, 0]
                    , "details":
                    [
                      "Stall-free write to memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:51"
              , "data":
              [110, 73, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":51
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"'__acl_barrier' Function Call"
                    , "data":
                    [110, 73, 0, 0]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:54"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":54
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:55"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":55
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:56"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":56
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:57"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":57
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:58"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":58
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:59"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":59
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:60"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":60
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
            , {
              "name":"reorder.cl:61"
              , "data":
              [9, 8, 0, 0]
              , "debug":
              [
                [
                  {
                    "filename":"/home/jbarrett/eece-6540-fall2017/Labs/Lab4/device/reorder.cl"
                    , "line":61
                  }
                ]
              ]
              , "subinfos":
              [
                {
                  "info":
                  {
                    "name":"Load"
                    , "data":
                    [9, 8, 0, 0]
                    , "details":
                    [
                      "Stall-free read from memory declared on reorder.cl:45."
                    ]
                  }
                  , "count":1
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}