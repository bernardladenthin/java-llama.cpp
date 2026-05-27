// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import org.jetbrains.kotlinx.lincheck.LinChecker;
import org.jetbrains.kotlinx.lincheck.annotations.Operation;
import org.jetbrains.kotlinx.lincheck.strategy.managed.modelchecking.ModelCheckingOptions;
import org.junit.jupiter.api.Test;

/**
 * Linearizability check for {@link CancellationToken}.
 *
 * <p>Verifies that concurrent {@link CancellationToken#cancel()} and
 * {@link CancellationToken#isCancelled()} invocations are linearizable:
 * every execution must be equivalent to some sequential execution of the
 * same operations.</p>
 */
public class CancellationTokenLincheckTest {

    private final CancellationToken token = new CancellationToken();

    @Operation
    public void cancel() {
        token.cancel();
    }

    @Operation
    public boolean isCancelled() {
        return token.isCancelled();
    }

    @Test
    public void modelCheckingTest() {
        ModelCheckingOptions options = new ModelCheckingOptions()
                .iterations(20)
                .invocationsPerIteration(500)
                .threads(2)
                .actorsPerThread(3);
        LinChecker.check(CancellationTokenLincheckTest.class, options);
    }
}
