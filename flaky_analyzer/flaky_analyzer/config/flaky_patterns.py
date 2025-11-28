"""
Flaky Test Pattern Definitions
==============================
Category mapping based on dataset analysis:
- Category 0: async wait (异步等待)
- Category 1: concurrency (并发)  
- Category 2: time (时间相关)
- Category 3: unordered collections (无序集合)
- Category 4: test order dependency (测试顺序依赖)
- Category 5: non-flaky (非flaky)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from enum import IntEnum

class FlakyCategory(IntEnum):
    ASYNC_WAIT = 0
    CONCURRENCY = 1
    TIME = 2
    UNORDERED_COLLECTIONS = 3
    TEST_ORDER_DEPENDENCY = 4
    NON_FLAKY = 5

CATEGORY_LABELS = {
    FlakyCategory.ASYNC_WAIT: "async wait",
    FlakyCategory.CONCURRENCY: "concurrency",
    FlakyCategory.TIME: "time",
    FlakyCategory.UNORDERED_COLLECTIONS: "unordered collections",
    FlakyCategory.TEST_ORDER_DEPENDENCY: "test order dependency",
    FlakyCategory.NON_FLAKY: "non-flaky"
}

@dataclass
class PatternConfig:
    """Configuration for non-determinism pattern detection"""
    
    # Category 0: Async Wait patterns
    async_wait_methods: List[str] = field(default_factory=lambda: [
        "Thread.sleep", "sleep", "wait", "await", "awaitTermination",
        "awaitUninterruptibly", "get", "join", "poll", "take",
        "acquire", "tryAcquire", "parkNanos", "parkUntil",
        "countDown", "await", "nodeHeartbeat", "dispatcher.await"
    ])
    
    async_wait_classes: List[str] = field(default_factory=lambda: [
        "CountDownLatch", "CyclicBarrier", "Semaphore", "Phaser",
        "Future", "CompletableFuture", "FutureTask", "Promise",
        "Condition", "Lock", "ReentrantLock", "DrainDispatcher"
    ])
    
    async_wait_patterns: List[str] = field(default_factory=lambda: [
        r"Thread\.sleep\s*\(",
        r"\.await\s*\(",
        r"\.wait\s*\(",
        r"\.get\s*\(\s*\d+.*TimeUnit",
        r"CountDownLatch",
        r"\.join\s*\(",
        r"CompletableFuture.*\.get\s*\(",
        r"assertEventually",
        r"waitFor",
        r"poll\s*\(",
        r"\.awaitTermination\s*\("
    ])
    
    # Category 1: Concurrency patterns  
    concurrency_classes: List[str] = field(default_factory=lambda: [
        "AtomicInteger", "AtomicLong", "AtomicBoolean", "AtomicReference",
        "Thread", "Runnable", "Callable", "Executor", "ExecutorService",
        "ThreadPoolExecutor", "ScheduledExecutorService", "ForkJoinPool",
        "ConcurrentHashMap", "ConcurrentLinkedQueue", "BlockingQueue",
        "ReentrantLock", "ReadWriteLock", "StampedLock", "Semaphore",
        "synchronized", "volatile"
    ])
    
    concurrency_patterns: List[str] = field(default_factory=lambda: [
        r"AtomicInteger|AtomicLong|AtomicBoolean|AtomicReference",
        r"new\s+Thread\s*\(",
        r"ExecutorService|Executor\.",
        r"synchronized\s*\(",
        r"\.incrementAndGet\s*\(",
        r"\.compareAndSet\s*\(",
        r"volatile\s+",
        r"Runnable|Callable",
        r"\.submit\s*\(",
        r"\.execute\s*\(",
        r"scheduleRecurring|schedule\s*\("
    ])
    
    # Category 2: Time patterns
    time_classes: List[str] = field(default_factory=lambda: [
        "Date", "Calendar", "LocalDate", "LocalTime", "LocalDateTime",
        "Instant", "ZonedDateTime", "OffsetDateTime", "Duration",
        "DateFormat", "SimpleDateFormat", "DateTimeFormatter",
        "System.currentTimeMillis", "System.nanoTime", "Clock"
    ])
    
    time_patterns: List[str] = field(default_factory=lambda: [
        r"new\s+Date\s*\(",
        r"System\.currentTimeMillis\s*\(",
        r"System\.nanoTime\s*\(",
        r"Calendar\.getInstance\s*\(",
        r"LocalDateTime\.now\s*\(",
        r"Instant\.now\s*\(",
        r"DateFormat\.",
        r"SimpleDateFormat",
        r"\.getTime\s*\(",
        r"getNanoTime\s*\(",
        r"TimeUnit\.",
        r"\.parse\s*\(.*[Dd]ate"
    ])
    
    # Category 3: Unordered Collections patterns
    unordered_collection_classes: List[str] = field(default_factory=lambda: [
        "HashMap", "HashSet", "Hashtable", "LinkedHashMap", "LinkedHashSet",
        "ConcurrentHashMap", "WeakHashMap", "IdentityHashMap",
        "EnumMap", "EnumSet"
    ])
    
    unordered_collection_patterns: List[str] = field(default_factory=lambda: [
        r"new\s+HashMap\s*[<(]",
        r"new\s+HashSet\s*[<(]",
        r"\.keySet\s*\(\s*\)",
        r"\.values\s*\(\s*\)",
        r"\.entrySet\s*\(\s*\)",
        r"for\s*\([^)]*:\s*\w+\.keySet\(\)",
        r"for\s*\([^)]*:\s*\w+\.values\(\)",
        r"\.iterator\s*\(\s*\)",
        r"ImmutableSet\.of\s*\(",
        r"assertEquals.*json|json.*assertEquals",
        r"containsExactlyInAnyOrder"
    ])
    
    # Category 4: Test Order Dependency patterns
    order_dependency_classes: List[str] = field(default_factory=lambda: [
        "File", "Path", "FileSystem", "FileInputStream", "FileOutputStream",
        "BufferedReader", "BufferedWriter", "RandomAccessFile",
        "Socket", "ServerSocket", "Connection", "DataSource",
        "Context", "ApplicationContext", "Configuration"
    ])
    
    order_dependency_patterns: List[str] = field(default_factory=lambda: [
        r"static\s+(?!final\s+\w+\s+\w+\s*=)",  # static non-final fields
        r"@BeforeClass|@AfterClass|@BeforeAll|@AfterAll",
        r"new\s+File\s*\(",
        r"Paths\.get\s*\(",
        r"Files\.",
        r"conf\.\w+\s*\(",
        r"getLocalPath",
        r"\.createNewFile\s*\(",
        r"\.delete\s*\(",
        r"\.mkdir\s*\(",
        r"LocalDirAllocator",
        r"testDir",
        r"Configuration\s+\w+\s*="
    ])

    # Assertion patterns to track
    assertion_patterns: List[str] = field(default_factory=lambda: [
        r"assert\w*\s*\(",
        r"Assert\.\w+\s*\(",
        r"assertThat\s*\(",
        r"assertEquals\s*\(",
        r"assertTrue\s*\(",
        r"assertFalse\s*\(",
        r"assertNull\s*\(",
        r"assertNotNull\s*\(",
        r"assertNotEquals\s*\(",
        r"assertSame\s*\(",
        r"assertThrows\s*\(",
        r"fail\s*\(",
        r"verify\s*\(",
        r"Matchers\.\w+"
    ])

# Pattern weights for confidence scoring
PATTERN_WEIGHTS = {
    FlakyCategory.ASYNC_WAIT: {
        "Thread.sleep": 0.9,
        "await": 0.85,
        "CountDownLatch": 0.8,
        "Future.get": 0.75,
        "poll": 0.6,
        "join": 0.7,
        "assertEventually": 0.95,
        "dispatcher.await": 0.85
    },
    FlakyCategory.CONCURRENCY: {
        "AtomicInteger": 0.8,
        "synchronized": 0.75,
        "Thread": 0.7,
        "Executor": 0.8,
        "incrementAndGet": 0.85,
        "scheduleRecurring": 0.9,
        "volatile": 0.7
    },
    FlakyCategory.TIME: {
        "System.currentTimeMillis": 0.9,
        "System.nanoTime": 0.9,
        "new Date": 0.8,
        "DateFormat": 0.75,
        "Instant.now": 0.85,
        "getNanoTime": 0.9,
        "Calendar.getInstance": 0.8
    },
    FlakyCategory.UNORDERED_COLLECTIONS: {
        "HashMap": 0.6,
        "HashSet": 0.6,
        "keySet": 0.7,
        "iterator": 0.65,
        "assertEquals.*json": 0.85,
        "ImmutableSet": 0.5,
        "containsExactlyInAnyOrder": 0.3  # This actually handles it properly
    },
    FlakyCategory.TEST_ORDER_DEPENDENCY: {
        "static": 0.6,
        "File": 0.5,
        "Configuration": 0.65,
        "@BeforeClass": 0.7,
        "testDir": 0.75,
        "LocalDirAllocator": 0.8
    }
}

# Non-deterministic operation categories
NONDETERMINISM_SOURCES = {
    "TIMING": ["sleep", "wait", "timeout", "delay", "poll"],
    "ASYNC": ["async", "await", "future", "promise", "callback", "CompletableFuture"],
    "CONCURRENCY": ["thread", "atomic", "synchronized", "lock", "executor", "concurrent"],
    "RANDOMNESS": ["random", "shuffle", "uuid", "Math.random"],
    "EXTERNAL_STATE": ["file", "database", "network", "socket", "connection", "config"],
    "TIME_DEPENDENT": ["date", "time", "timestamp", "nanoTime", "currentTimeMillis"],
    "ORDER_SENSITIVE": ["hashmap", "hashset", "iterator", "keySet", "entrySet"]
}
